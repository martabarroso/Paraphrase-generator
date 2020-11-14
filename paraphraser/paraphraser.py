from .translator import Translator
from typing import List, Optional, Set, Dict
from .constants import SYSTEM_LANGUAGE
from collections import OrderedDict
from copy import deepcopy


class Paraphraser:
    def __init__(self, translators: List[str]):
        self.translators = {}
        for translator_name in translators:
            translator = Translator.build(translator_name)
            for direction in translator.directions:
                orig, target = direction
                if orig != SYSTEM_LANGUAGE:
                    continue
                if orig in self.translators:
                    if target in self.translators[orig]:
                        self.translators[orig][target][translator_name] = translator
                    else:
                        self.translators[orig][target] = {translator_name: translator}
                else:
                    self.translators[orig] = {target: {translator_name: translator}}

    def paraphrase(self, sentence: str, n_paraphrases: int) -> List[str]:
        return self.paraphrase_sentences([sentence], n_paraphrases)[0]

    def _paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> List[List[str]]:
        raise NotImplementedError()

    def paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> List[List[str]]:
        for idx, sentence in enumerate(sentences):
            sentences[idx] = sentence.strip()
        return self._paraphrase_sentences(sentences, n_paraphrases_per_sentence)

    @staticmethod
    def build(method: str, translators: List[str]):
        if method == 'dummy':
            import logging
            logging.warning('Using DUMMY paraphraser')
            return DummyParaphraser()
        elif method == 'roundtrip':
            return RoundTripParaphraser(translators)
        elif method == 'intermediate':
            return IntermediateParaphraser(translators)
        else:
            return TwoCyclesParaphraser(translators)


class DummyParaphraser:

    def _paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> Dict[str, List]:
        # Identity
        result = OrderedDict()
        for sentence in sentences:
            result[sentence.strip()] = [sentence.strip()]*n_paraphrases_per_sentence
        return result


class RoundTripParaphraser(Paraphraser):
    def _paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> Dict[str, List]:
        # English -> Other language 1 -> English
        n_translations_per_sentence = n_paraphrases_per_sentence  # TODO: Check
        sentence_dict = OrderedDict()
        for sentence in sentences:
            sentence_dict[sentence] = set()
        first_trip = {}
        for from_orig_to_other in self.translators[SYSTEM_LANGUAGE]:
            first_trip[from_orig_to_other] = deepcopy(sentence_dict)
            for system_name in self.translators[SYSTEM_LANGUAGE][from_orig_to_other]:
                translated_sentences = \
                    self.translators[SYSTEM_LANGUAGE][from_orig_to_other][system_name].translate_sentences(sentences,
                                                                                            n_translations_per_sentence)
                assert len(translated_sentences) == len(sentences)
                for idx, translations in enumerate(translated_sentences):
                    assert len(translations) == n_translations_per_sentence  # TODO: Check, n_translations_per_sentence
                    first_trip[from_orig_to_other][list(first_trip[from_orig_to_other].keys())[idx]].update(
                        translations)
        result = deepcopy(sentence_dict)
        for from_other_to_orig in first_trip:
            for system_name in self.translators[SYSTEM_LANGUAGE][from_other_to_orig]:
                batched_sentences = []
                batched_idx = []
                idx = 0
                for sentence in sentences:
                    batched_sentences.extend(first_trip[from_other_to_orig][sentence])
                    batched_idx.append((idx, idx+len(first_trip[from_other_to_orig][sentence])))
                    idx += len(first_trip[from_other_to_orig][sentence])

                backtranslated_sentences = \
                    self.translators[SYSTEM_LANGUAGE][from_other_to_orig][system_name].translate_sentences(
                        batched_sentences, n_translations_per_sentence)

                for idx, sentence in enumerate(sentences):
                    i, j = batched_idx[idx]
                    for paraphrasis in backtranslated_sentences[i:j][0]:  # TODO: Check backtranslated_sentences[i:j][0]
                        result[list(result.keys())[idx]].add(paraphrasis)
        for sentence in sentences:
            result[sentence] = list(result[sentence])
        # TODO: Should n_paraphrases_per_sentence be used to throw an error if less paraphrases are generated?
        # Should it be used to cut as a maximum number?
        # I don't think so...
        return result


class IntermediateParaphraser(Paraphraser):
    def _paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> Dict[str, List]:
        # TODO: English -> Other language 1 -> Other language 2 -> English
        pass


class TwoCyclesParaphraser(Paraphraser):
    def _paraphrase_sentences(self, sentences: List[str], n_paraphrases_per_sentence: int) -> Dict[str, List]:
        # TODO: English -> Other language 1 -> English -> Other language 2 -> English
        pass
