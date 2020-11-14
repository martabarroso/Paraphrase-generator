from .translator import Translator
from typing import List, Optional
from .constants import LANGUAGE


class Paraphraser:
    def __init__(self, translators: List[str]):
        self.translators = {}
        for t in translators:
            translator = Translator.build(t)
            for direction in translator.directions:
                if direction in self.translators:
                    self.translators[direction][t] = translator
                else:
                    self.translators[direction] = {t: translator}

    def paraphrase(self, sentence: str, n_paraphrases: int) -> List[str]:
        return self.paraphrase_sentences([sentence], n_paraphrases)[0]

    def paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> List[List[str]]:
        raise NotImplementedError()

    @staticmethod
    def build(method: str, translators: List[str]):
        if method == 'dummy':
            import logging
            logging.warning('Using DUMMY paraphraser')
            return DummyParaphraser()
        elif method == 'roundtrip':
            assert len(translators) in [1, 2]  # Either bidirectional system, or two systems
            return RoundTripParaphraser(translators)
        elif method == 'intermediate':
            return IntermediateParaphraser(translators)
        else:
            return TwoCyclesParaphraser(translators)


class DummyParaphraser:

    def paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: Optional[int]) -> List[List[str]]:
        # Identity
        return [[sentence.strip()]*n_paraphrases_per_sentence for sentence in sentences]


class RoundTripParaphraser(Paraphraser):
    def paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: Optional[int]) -> List[List[str]]:
        # English -> Other language 1 -> English
        # TODO: Check
        translated_sentences = []
        for orig_to_other_system in self.translators[LANGUAGE]:
            translated_sentences.append(orig_to_other_system.translate(sentences, n_paraphrases_per_sentence))
        result = [[] for _ in range(len(sentences))]
        for i in range(len(sentences)):
            result[i] = []
            for translations in translated_sentences:
                result[i].append(translations[i].strip())
        return result


class IntermediateParaphraser(Paraphraser):
    def paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> List[List[str]]:
        # TODO: English -> Other language 1 -> Other language 2 -> English
        pass


class TwoCyclesParaphraser(Paraphraser):
    def paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> List[List[str]]:
        # TODO: English -> Other language 1 -> English -> Other language 2 -> English
        pass
