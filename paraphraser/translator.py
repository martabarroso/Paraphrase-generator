from typing import List, Tuple, Union
import torch
from .constants import BEAM
import logging
import ssl
from .utils import normalize_spaces_remove_urls
from glob import glob
import numpy as np
from scipy.io.wavfile import write
import uuid
import os


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
        elif translator_name == 'nvidia_tts_en':
            return NVIDIATTS()
        elif translator_name == 'silero_asr_en':
            return SileroASR()
        else:
            raise NotImplementedError(translator_name)

    @property
    def directions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def _translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        raise NotImplementedError()

    def translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(sentences, str):
            sentences = normalize_spaces_remove_urls(sentences)
        else:
            sentences = list(map(normalize_spaces_remove_urls, sentences))
        return self._translate_sentences(sentences)

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


class SileroASR(Translator):

    def __init__(self):
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

        for speech_file in speech_files:
            pass#os.remove(speech_file)
        if isinstance(sentences, str):
            res = res[0]
        return res


class NVIDIATTS(Translator):

    def __init__(self):
        super().__init__()
        ssl._create_default_https_context = ssl._create_unverified_context

        cuda = torch.cuda.is_available()

        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        if not cuda:
            logging.warning('Running on CPU')

        if cuda:
            self.waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
            self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
            self.waveglow = self.waveglow.to('cuda')
            self.tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
            self.tacotron2 = self.tacotron2.to('cuda')

        else:  # These NVIDIA entries do not support mapping to CPU. We have to do it manually
            os.makedirs('resources', exist_ok=True)
            import gdown
            file_id_tacotron = '1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA'
            url_tacotron = "https://drive.google.com/uc?id={}".format(file_id_tacotron)
            tacotron_checkpoint = os.path.join('resources', 'tacotron2_statedict.pt')
            if not os.path.exists(tacotron_checkpoint):
                gdown.download(url_tacotron, tacotron_checkpoint, quiet=False)
            tacotron_checkpoint = torch.load(tacotron_checkpoint, map_location="cpu")

            file_id_waveglow = '1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx'
            url_waveglow = "https://drive.google.com/uc?id={}".format(file_id_waveglow)
            waveglow_checkpoint = os.path.join('resources', 'waveglow_statedict.pt')
            if not os.path.exists(waveglow_checkpoint):
                gdown.download(url_waveglow, waveglow_checkpoint, quiet=False)

            waveglow_checkpoint = torch.load(waveglow_checkpoint, map_location="cpu")

            state_dict_tacotron = {
                key.replace("module.", ""): value for key, value in tacotron_checkpoint["state_dict"].items()}
            # Apply the state dict to the model
            self.tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2',
                                            pretrained=False)
            self.tacotron2.load_state_dict(state_dict_tacotron)

            self.waveglow = waveglow_checkpoint['model']

        self.tacotron2.eval()
        self.waveglow.eval()

        self.name = 'nvidia_tts_en'
        self._directions = [('en', 'en_speech')]

    @property
    def directions(self) -> List[Tuple[str, str]]:
        return self._directions

    def _translate_sentences(self, sentences: Union[List[str], str]) -> Union[List[str], str]:
        one_sentence = False
        if isinstance(sentences, str):
            one_sentence = True
            sentences = [sentences]
        translated = []
        os.makedirs('tmp', exist_ok=True)
        for text in sentences:
            # preprocessing
            sequence = np.array(self.tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
            sequence = torch.from_numpy(sequence).to(device=self.device, dtype=torch.int64)

            # run the models
            with torch.no_grad():
                _, mel, _, _ = self.tacotron2.infer(sequence)
                audio = self.waveglow.infer(mel)
            audio_numpy = audio[0].data.cpu().numpy()
            rate = 22050
            filename = uuid.uuid4().hex
            filename = os.path.join('tmp', f"{filename}.wav")
            write(filename, rate, audio_numpy)
            translated.append(filename)

        if one_sentence:
            translated = translated[0]
        return translated


def speech2text():

    device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language='en', # also available 'de', 'es'
                                           device=device)
    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils  # see function signature for details

    # download a single file, any format compatible with TorchAudio (soundfile backend)
    torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
                                   dst ='speech_orig.wav', progress=True)
    test_files = glob('speech_orig.wav')
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)

    output = model(input)
    for example in output:
        print(decoder(example.cpu()))

def text2speech():
    device = "cpu"
    import torch
    #waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow', map_location=torch.device('cpu'))
    #waveglow = waveglow.remove_weightnorm(waveglow)
    #waveglow = waveglow.to('cuda')
    #waveglow.eval()
    '''
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2 = tacotron2.to('cuda')
    tacotron2.eval()
    '''
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', pretrained=False)

    checkpoint = torch.load('/home/jordiarmigol/Downloads/tacotron2_statedict.pt', map_location="cpu")

    #checkpoint = torch.hub.load_state_dict_from_url(
    #    'https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/1/files/nvidia_tacotron2pyt_fp32_20190306.pth',
    #    map_location="cpu")
    #" https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_fp32/versions/19.09.0"

    # Unwrap the DistributedDataParallel module
    # module.layer -> layer
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

    # Apply the state dict to the model
    tacotron2.load_state_dict(state_dict)
    text = "hello world, I missed you"
    # preprocessing
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    #sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
    sequence = torch.from_numpy(sequence).to(dtype=torch.int64)

    # run the models
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
        audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050
    write("audio.wav", rate, audio_numpy)