from typing import List, Tuple, Union
import torch
from .constants import BEAM, BATCH
import logging
import ssl
from .utils import normalize_spaces_remove_urls
from glob import glob
import numpy as np
import uuid
import os
import yaml
from transformers import MarianTokenizer, MarianMTModel
from tacotron_pytorch.src.module import Tacotron
from tacotron_pytorch.src.symbols import txt2seq
from tacotron_pytorch.src.utils import AudioProcessor
from tqdm import tqdm
import datetime


class Translator:
    def __init__(self):
        self.beam = BEAM

    @staticmethod
    def build(translator_name: str):
        if translator_name == 'fair-wmt19-en-de':
            return FAIRPretrainedWMT19EnglishGermanTranslator()
        elif translator_name == 'fair-wmt19-de-en':
            return FAIRPretrainedWMT19GermanEnglishTranslator()
        elif translator_name == 'fair-wmt19-en-ru':
            return FAIRPretrainedWMT19EnglishRussianTranslator()
        elif translator_name == 'fair-wmt19-ru-en':
            return FAIRPretrainedWMT19RussianEnglishTranslator()
        elif translator_name == 'fair-wmt20-en-ta':
            return FAIRPretrainedWMT20EnglishTamilTranslator()
        elif translator_name == 'fair-wmt20-ta-en':
            return FAIRPretrainedWMT20TamilEnglishTranslator()
        elif translator_name == 'silero_asr_en':
            return SileroASR()
        elif translator_name == 'tacotron_pytorch':
            return TacotronPyTorch()
        elif translator_name.startswith('marian'):
            split = translator_name.split('-')
            assert len(split) == 3
            assert split[0] == 'marian'
            assert len(split[1]) == 2
            assert len(split[2]) == 2
            return MarianHFTranslator(src_lang=split[1], tgt_lang=split[2])
        else:
            raise NotImplementedError(translator_name)

    @property
    def directions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def _translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        raise NotImplementedError()

    def translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        logging.info('Translating...')
        t0 = datetime.datetime.now().timestamp()
        if isinstance(sentences, str):
            sentences = normalize_spaces_remove_urls(sentences)
        else:
            sentences = list(map(normalize_spaces_remove_urls, sentences))

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        if len(sentences) > BATCH:
            res = []
            for sentence_batch in tqdm(list(chunks(sentences, BATCH))):
                res.extend(self._translate_sentences(sentence_batch))
        else:
            res = self._translate_sentences(sentences)
        t1 = datetime.datetime.now().timestamp()
        logging.info(f'Translated: Elapsed {t1 - t0}s')
        return res

    def translate_one_sentence(self, sentence: str) -> str:
        raise self.translate_sentences([sentence])[0]


class FAIRHubTranslator(Translator):
    def __init__(self, hub_entry: str, name: str, directions: List[Tuple[str, str]]):
        super().__init__()
        ssl._create_default_https_context = ssl._create_unverified_context
        self.system = torch.hub.load('pytorch/fairseq', hub_entry)
        self.system.eval()
        if torch.cuda.is_available():
            self.system.cuda()
        else:
            logging.warning('Running on CPU')
        self.name = name
        self._directions = directions

    @property
    def directions(self) -> List[Tuple[str, str]]:
        return self._directions

    def _translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        translated = self.system.translate(sentences, beam=self.beam)
        return translated


class FAIRPretrainedWMT19EnglishGermanTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.en-de.single_model', 'fair-wmt19-en-de', [('en', 'de')])


class FAIRPretrainedWMT19GermanEnglishTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.de-en.single_model', 'fair-wmt19-de-en', [('de', 'en')])


class FAIRPretrainedWMT19EnglishRussianTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.en-ru.single_model', 'fair-wmt19-en-ru', [('en', 'ru')])


class FAIRPretrainedWMT19RussianEnglishTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.ru-en.single_model', 'fair-wmt19-ru-en', [('ru', 'en')])


class FAIRPretrainedWMT20EnglishTamilTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt20.en-ta', 'fair-wmt20-en-ta', [('en', 'ta')])


class FAIRPretrainedWMT20TamilEnglishTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt20.ta-en', 'fair-wmt20-ta-en', [('ta', 'en')])


class SileroASR(Translator):

    def __init__(self, remove_tmp: bool = True):
        super().__init__()
        ssl._create_default_https_context = ssl._create_unverified_context

        cuda = torch.cuda.is_available()

        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        if not cuda:
            logging.warning('Running on CPU')

        self.model, self.decoder, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                              model='silero_stt',
                                                              language='en',  # also available 'de', 'es'
                                                              device=self.device)
        self.model.eval()

        self.name = 'silero_asr_en'
        self._directions = [('en_speech', 'en')]

        self.remove_tmp = remove_tmp

    @property
    def directions(self) -> List[Tuple[str, str]]:
        return self._directions

    def _translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        (read_batch, split_into_batches,
         read_audio, prepare_model_input) = self.utils  # see function signature for details

        def flatten(l):
            flat = []
            for e in l:
                flat.extend(e)
            return flat

        speech_files = glob(sentences) if isinstance(sentences, str) else flatten(
            [glob(sentence) for sentence in sentences])

        batches = split_into_batches(speech_files, batch_size=10)
        input_ = prepare_model_input(read_batch(batches[0]), device=self.device)

        output = self.model(input_)
        res = []
        for example in output:
            res.append(self.decoder(example.cpu()))

        if self.remove_tmp:
            for speech_file in speech_files:
                os.remove(speech_file)
        if isinstance(sentences, str):
            res = res[0]
        return res


class TacotronPyTorch(Translator):
    def __init__(self):
        super().__init__()
        config = os.path.join('tacotron_pytorch', 'config', 'config.yaml')
        self.config = yaml.load(open(config, 'r'))
        checkpoint = os.path.join('tacotron_pytorch', 'ckpt', 'checkpoint_step138000.pth')
        cuda = torch.cuda.is_available()

        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.model = self.load_ckpt(self.config, checkpoint, self.device)
        if not cuda:
            logging.warning('Running on CPU')
        else:
            self.model.to('cuda')

        self.name = 'tacotron_pytorch'
        self._directions = [('en', 'en_speech')]

    @property
    def directions(self) -> List[Tuple[str, str]]:
        return self._directions

    def _translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        one_sentence = False
        if isinstance(sentences, str):
            one_sentence = True
            sentences = [sentences]
        os.makedirs('tmp', exist_ok=True)
        res = []
        for text in sentences:
            seq = np.asarray(txt2seq(text))
            seq = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            # Decode
            with torch.no_grad():
                mel, spec, attn = self.model(seq)
            # Generate wav file
            spec = spec.cpu()
            ap = AudioProcessor(**self.config['audio'])
            wav = ap.inv_spectrogram(spec[0].numpy().T)
            filename = uuid.uuid4().hex
            filename = os.path.join('tmp', f"{filename}.wav")
            ap.save_wav(wav, filename)
            res.append(filename)
        if one_sentence:
            res = res[0]
        return res

    @staticmethod
    def load_ckpt(config, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        model = Tacotron(**config['model']['tacotron'])
        model.load_state_dict(ckpt['state_dict'])
        # This yields the best performance, not sure why
        # model.mel_decoder.eval()
        model.encoder.eval()
        model.postnet.eval()
        return model


class MarianHFTranslator(Translator):
    def __init__(self, src_lang: str, tgt_lang: str):
        super(MarianHFTranslator).__init__()
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        cuda = torch.cuda.is_available()

        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        if not cuda:
            logging.warning('Running on CPU')
        else:
            self.model.to('cuda')

        self._directions = [(src_lang, tgt_lang)]

    @property
    def directions(self) -> List[Tuple[str, str]]:
        return self._directions

    def _translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        one_sentence = False
        if isinstance(sentences, str):
            one_sentence = True
            sentences = [sentences]
        to_translate = self.tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt").to(self.device)
        translated = self.model.generate(**to_translate)
        tgt_text = [self.tokenizer.decode(t.cpu(), skip_special_tokens=True) for t in translated]
        if one_sentence:
            tgt_text = tgt_text[0]
        return tgt_text
