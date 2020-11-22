import json
from typing import List, Dict

from evaluation.configuration import CONFIGURATION
from evaluation.model import TextClassifier
from evaluation.preprocessing import Preprocessing
from evaluation.run import Run


class Evaluator:
    @staticmethod
    def get_all_evaluators():
        return [InstrinsicEvaluator(), ExtrinsicEvaluator()]

    def evaluate_paraphrases(self, sentences2paraphrases_dict: Dict) -> Dict:
        raise NotImplementedError()


class InstrinsicEvaluator:

    def evaluate_individual_sentence(self, original_sentence: str, generated_paraphrases: List[str]) -> Dict:
        # TODO: This should compute metrics
        raise NotImplementedError()

    def evaluate_paraphrases(self, sentences2paraphrases_dict: Dict) -> Dict:
        results = []
        for sentence, paraphrases in sentences2paraphrases_dict.items():
            results.append(self.evaluate_individual_sentence(sentence, paraphrases))
        # TODO: aggregate results into dict? (E.g., average, std...)
        return {}


class ExtrinsicEvaluator:

    def evaluate_paraphrases(self, original_sentence: str, generated_paraphrases: List[str]) -> Dict:
        # TODO: Train ../evaluation/sentence_classifier with paraphrases as data augmentation and compare
        # the results with the baseline

        with open('../output/example_output/paraphrases.json') as json_file:
            paraphrases = json.load(json_file)

        # TODO: Convert data into the format: sentence, class
        input_path = ''
        data = Preprocessing(CONFIGURATION['num_words'], CONFIGURATION['seq_len'], input_path).preprocess()
        model = TextClassifier(CONFIGURATION)
        Run().train(model, data, CONFIGURATION)

        raise NotImplementedError()
