import json
from typing import List, Dict
import pandas as pd
import nltk
import pylev
from sentence_transformers import SentenceTransformer, util

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

    @staticmethod
    def evaluate_individual_sentence(original_sentence, paraphrase) -> Dict:
        original_sentence_list = original_sentence.split()
        paraphrase_list = paraphrase.split()

        # Bleu score
        bleu_score = nltk.translate.bleu_score.sentence_bleu([original_sentence], paraphrase)

        # cosine similarity
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        emb1 = model.encode(original_sentence)
        emb2 = model.encode(paraphrase)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)

        # Levenshtein distance
        edit_distance = pylev.levenshtein(original_sentence_list, paraphrase_list)

        metrics = {'bleu_score': bleu_score, 'cosine_similarity': cos_sim.numpy()[0][0], 'edit_distance': edit_distance,
                   'cos_edit_distance': cos_sim.numpy()[0][0] * edit_distance}

        return metrics

    def evaluate_paraphrases(self, sentences2paraphrases_dict: Dict) -> Dict:
        results = {'bleu_score': [], 'cosine_similarity': [], 'edit_distance': [], 'cos_edit_distance': [],
                   'original_sentence': [], 'paraphrase': []}

        for sentence, paraphrases in sentences2paraphrases_dict.items():
            for paraphrase in paraphrases:
                result = self.evaluate_individual_sentence(sentence, paraphrase)
                results['bleu_score'].append(result['bleu_score'])
                results['cosine_similarity'].append(result['cosine_similarity'])
                results['edit_distance'].append(result['edit_distance'])
                results['cos_edit_distance'].append(result['cos_edit_distance'])
                results['original_sentence'].append(sentence)
                results['paraphrase'].append(paraphrases)

        df = pd.DataFrame.from_dict(results)
        statistics = df.describe(include='all').to_dict()
        return statistics


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
