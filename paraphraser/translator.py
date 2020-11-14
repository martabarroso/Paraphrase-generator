from typing import List, Tuple, Union
import torch
from .constants import BEAM
import logging


class Translator:
    def __init__(self):
        # TODO: Investigate if this should be a parameter to generate more translations.
        # Otherwise, 5 is a reasonable default.
        self.beam = BEAM

    @staticmethod
    def build(translator_name: str):
        if translator_name == 'fair-wmt19-en-de':
            return FAIRPretrainedWMT19EnglishGermanTranslator()
        elif translator_name == 'fair-wmt19-de-en':
            return FAIRPretrainedWMT19GermanEnglishTranslator()
        else:
            raise NotImplementedError(translator_name)

    @property
    def directions(self) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def translate_sentences(self, sentences: Tuple[List[str], str], n_translations: int) -> Tuple[List[List[str]],
                                                                                                  List[str]]:
        raise NotImplementedError()

    def translate_one_sentence(self, sentence: str, n_translations: int) -> List[str]:
        raise self.translate([sentence], n_translations)[0]


class FAIRHubTranslator(Translator):
    def __init__(self, hub_entry: str, name: str, directions: List[Tuple[str, str]]):
        super().__init__()
        self.system = torch.hub.load('pytorch/fairseq', hub_entry)
        self.name = name
        self._directions = directions

    @property
    def directions(self) -> List[Tuple[str, str]]:
        return self._directions

    def translate_sentences(self, sentences: Union[List[str], str], n_translations: int) -> Union[List[List[str]],
                                                                                                  List[str]]:

        if n_translations != 1:
            # raise ValueError(f'{self.name} can only translate into one sentence')  # TODO: Investigate how to sample
            logging.warning(f'{self.name} can only translate into one sentence (repeating translations to simulate it)')

        translated = self.system.translate(sentences, beam=self.beam)

        # Hack to simulate that the system is actually returning a sample of sentences. TODO: Fix.
        if not isinstance(translated, list):
            return [translated]*n_translations
        result = []
        for sentence in translated:
            result.append([sentence]*n_translations)
        return result


class FAIRPretrainedWMT19EnglishGermanTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.en-de.single_model', 'fair-wmt19-en-de', [('en', 'de')])


class FAIRPretrainedWMT19GermanEnglishTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.de-en.single_model', 'fair-wmt19-de-en', [('de', 'en')])
