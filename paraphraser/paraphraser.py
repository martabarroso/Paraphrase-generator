from .translator import Translator
from typing import List, Dict
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
                    pass
                if orig in self.translators:
                    if target in self.translators[orig]:
                        self.translators[orig][target][translator_name] = translator
                    else:
                        self.translators[orig][target] = {translator_name: translator}
                else:
                    self.translators[orig] = {target: {translator_name: translator}}

    def paraphrase(self, sentence: str) -> List[str]:
        res = self.paraphrase_sentences([sentence])
        for sentence in res:
            return res[sentence]

    def _paraphrase_sentences(self, sentences: List[str]) -> Dict[str, List[str]]:
        raise NotImplementedError()

    def paraphrase_sentences(self, sentences: List[str]) -> Dict[str, List[str]]:
        for idx, sentence in enumerate(sentences):
            sentences[idx] = sentence.strip()
        paraphrases = self._paraphrase_sentences(sentences)
        for original_sentence in paraphrases:
            paraphrases[original_sentence] = list(set(paraphrases[original_sentence]))
        return paraphrases

    def _translate_path(self, sentences: List[str], lang_path_orig: str, lang_path_via: str,
                        lang_path_dest: str) -> Dict[str, List]:
        # English -> Other language 1 -> English
        # This implementation has the advantage f considering that more than one translator
        # for a given direction might exist, and than, for returning to the original language, more than one path might
        # exist (e.g., English -> Other language via EN-DE translator 1, and Other language to English via DE-EN
        # translator 3), and it handles batched translation.

        sentence_dict = OrderedDict()
        for sentence in sentences:
            sentence_dict[sentence] = set()
        first_trip = {}
        for from_orig_to_other in self.translators[lang_path_orig]:
            if lang_path_via != 'all' and from_orig_to_other != lang_path_via:
                continue
            first_trip[from_orig_to_other] = deepcopy(sentence_dict)
            for system_name in self.translators[lang_path_orig][from_orig_to_other]:
                translated_sentences = \
                    self.translators[lang_path_orig][from_orig_to_other][system_name].translate_sentences(sentences)
                assert len(translated_sentences) == len(sentences)
                for idx, translations in enumerate(translated_sentences):
                    assert isinstance(translations, str)
                    first_trip[from_orig_to_other][list(first_trip[from_orig_to_other].keys())[idx]].add(
                        translations)
        result = deepcopy(sentence_dict)
        for from_other_to_orig in first_trip:
            for system_name in self.translators[from_other_to_orig][lang_path_dest]:
                batched_sentences = []
                batched_idx = []
                idx = 0
                for sentence in sentences:
                    batched_sentences.extend(first_trip[from_other_to_orig][sentence])
                    batched_idx.append((idx, idx+len(first_trip[from_other_to_orig][sentence])))
                    idx += len(first_trip[from_other_to_orig][sentence])

                backtranslated_sentences = \
                    self.translators[from_other_to_orig][lang_path_dest][system_name].translate_sentences(
                        batched_sentences)

                for idx, sentence in enumerate(sentences):
                    i, j = batched_idx[idx]
                    for paraphrasis in backtranslated_sentences[i:j]:
                        result[list(result.keys())[idx]].add(paraphrasis)
        for sentence in sentences:
            result[sentence] = list(result[sentence])
        return result

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
            return NCyclesParaphraser(translators)


class DummyParaphraser:

    def _paraphrase_sentences(self, sentences: List[str]) -> Dict[str, List]:
        # Identity
        result = OrderedDict()
        for sentence in sentences:
            result[sentence.strip()] = [sentence.strip()]
        return result


class RoundTripParaphraser(Paraphraser):
    def _paraphrase_sentences(self, sentences: List[str]) -> Dict[str, List]:
        non_system_languages = []
        for language in self.translators[SYSTEM_LANGUAGE]:
            non_system_languages.append(language)
            if SYSTEM_LANGUAGE not in self.translators[language]:
                raise RuntimeError(f"Direction {language} -> {SYSTEM_LANGUAGE} not available")
        result = OrderedDict()
        for language in non_system_languages:
            paraphrases = self._translate_path(sentences, lang_path_orig=SYSTEM_LANGUAGE, lang_path_via=language,
                                               lang_path_dest=SYSTEM_LANGUAGE)
            for original_sentence in paraphrases:
                if original_sentence in result:
                    if paraphrases[original_sentence] not in result[original_sentence]:
                        result[original_sentence].extend(paraphrases[original_sentence])
                else:
                    result[original_sentence] = paraphrases[original_sentence]
        return result


class IntermediateParaphraser(Paraphraser):
    def _paraphrase_sentences(self, sentences: List[str]) -> Dict[str, List]:
        # TODO (Future work): English -> Other language 1 -> Other language 2 -> English
        raise NotImplementedError()


class NCyclesParaphraser(Paraphraser):
    def __init__(self, *args, n_cycles: int = 2):
        super().__init__(*args)
        self.n_cycles = n_cycles

    def _paraphrase_sentences(self, sentences: List[str]) -> Dict[str, List]:
        # English -> Other language 1 -> English -> Other language 2 -> English

        current_paraphrases = self._translate_path(sentences, lang_path_orig=SYSTEM_LANGUAGE, lang_path_via='all',
                                                   lang_path_dest=SYSTEM_LANGUAGE)

        def flatten_one_level(x):
            res = []
            for e in x:
                res.extend(e)
            return res

        def update_paraphrases(old, new):
            for original_sentence in old:
                new_paraphrases_ = []
                for old_paraphrase in old[original_sentence]:
                    new_paraphrases_.extend(new[old_paraphrase])
                old[original_sentence] = new_paraphrases_

        previous_sentences = flatten_one_level(list(current_paraphrases.values()))
        for cycle in range(1, self.n_cycles):

            new_paraphrases = self._translate_path(previous_sentences, lang_path_orig=SYSTEM_LANGUAGE,
                                                   lang_path_via='all',
                                                   lang_path_dest=SYSTEM_LANGUAGE)

            update_paraphrases(current_paraphrases, new_paraphrases)
            previous_sentences = flatten_one_level(list(current_paraphrases.values()))

        return current_paraphrases
