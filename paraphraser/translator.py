from typing import List
import torch


class Translator:
    def __init__(self):
        # TODO: Investigate if this should be a parameter to generate more translations.
        # Otherwise, 5 is a reasonable default.
        self.beam = 5

    @staticmethod
    def build(translator_name: str):
        if translator_name == 'fair-wmt19-en-de':
            return FAIRPretrainedWMT19EnglishGermanTranslator()
        elif translator_name == 'fair-wmt19-de-en':
            return FAIRPretrainedWMT19GermanEnglishTranslator()
        else:
            raise NotImplementedError(translator_name)

    @property
    def directions(self) -> List[str]:
        raise NotImplementedError()

    def _translate(self, sentence: str, n_translations: int) -> List[str]:
        raise NotImplementedError()

    def _batched_translate(self, sentences: List[str], n_translations: int) -> List[List[str]]:
        # If the translation system support batched translation, use it for efficiency
        raise NotImplementedError()

    def translate_sentences(self, sentences: List[str], n_translations: int) -> List[List[str]]:
        try:
            return self._batched_translate(sentences, n_translations)
        except NotImplementedError:
            result = []
            for sentence in sentences:
                result.append(self._translate(sentence, n_translations))
            return result

    def translate_one_sentence(self, sentence: str, n_translations: int) -> List[str]:
        raise self._translate(sentence, n_translations)


class FAIRHubTranslator(Translator):
    def __init__(self, hub_entry: str, name: str, directions: List[str]):
        super().__init__()
        self.system = torch.hub.load('pytorch/fairseq', hub_entry)
        self.name = name
        self._directions = directions

    @property
    def directions(self) -> List[str]:
        return self._directions

    def _translate(self, sentence: str, n_translations: int) -> List[str]:
        if n_translations != 1:
            raise ValueError(f'{self.name} can only translate into one sentence')  # TODO: Investigate how to sample
        return [self.system(sentence, beam=self.beam)]

    def _batched_translate(self, sentences: List[str], n_translations: int) -> List[List[str]]:
        if n_translations != 1:
            raise ValueError(f'{self.name} can only translate into one sentence')  # TODO: Investigate how to sample
        return [self.system(sentences, beam=self.beam)]


class FAIRPretrainedWMT19EnglishGermanTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.en-de.single_model', 'fair-wmt19-en-de', ['de2en'])


class FAIRPretrainedWMT19GermanEnglishTranslator(FAIRHubTranslator):
    def __init__(self):
        super().__init__('transformer.wmt19.de-en.single_model', 'fair-wmt19-de-en', ['en2de'])
