from typing import Dict
import pandas as pd
import nltk
import pylev
import os
from sentence_transformers import SentenceTransformer, util
from .utils import normalize_spaces_remove_urls, deterministic

from extrinsic_evaluation.configuration import CONFIGURATION
from extrinsic_evaluation.model import TextClassifier
from extrinsic_evaluation.preprocessing import Preprocessing
from extrinsic_evaluation.run import Run
import logging


nltk.download('punkt')


class Evaluator:
    @staticmethod
    def get_all_evaluators():
        return [IntrinsicEvaluator(), ExtrinsicEvaluator()]

    def evaluate_paraphrases(self, sentences2paraphrases_dict: Dict) -> Dict:
        raise NotImplementedError()


class IntrinsicEvaluator:

    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('stsb-distilbert-base')

    def evaluate_individual_sentence(self, original_sentence, paraphrase) -> Dict:

        original_sentence_tokens = nltk.word_tokenize(normalize_spaces_remove_urls(original_sentence))
        paraphrase_tokens = nltk.word_tokenize(normalize_spaces_remove_urls(paraphrase))

        # Bleu score
        bleu_score = nltk.translate.bleu_score.sentence_bleu([normalize_spaces_remove_urls(original_sentence)],
                                                             normalize_spaces_remove_urls(paraphrase))

        # Sentence embedding cosine similarity
        emb1 = self.model.encode(original_sentence)
        emb2 = self.model.encode(paraphrase)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)

        # Levenshtein distance
        edit_distance = pylev.levenshtein(original_sentence_tokens, paraphrase_tokens)
        length = max(len(original_sentence_tokens), len(paraphrase_tokens))
        normalized_edit_distance = (length - edit_distance)/length

        # Jaccard
        jaccard = nltk.jaccard_distance(set(original_sentence_tokens), set(paraphrase_tokens))

        # Jaccard * cosine similarity
        jaccard_embedding_factor = jaccard*cos_sim.item()

        metrics = {'original_sentence': original_sentence,
                   'paraphrase': paraphrase, 'bleu_score': bleu_score,
                   'normalized_original_sentence': normalize_spaces_remove_urls(original_sentence),
                   'normalized_paraphrase': normalize_spaces_remove_urls(paraphrase),
                   'embedding_cosine_similarity': cos_sim.item(), 'edit_distance': edit_distance,
                   'normalized_edit_distance': normalized_edit_distance, 'jaccard': jaccard,
                   'jaccard_embedding_factor': jaccard_embedding_factor}

        return metrics

    def evaluate_paraphrases(self, sentences2paraphrases_dict: Dict) -> Dict:
        results = {'bleu_score': [], 'embedding_cosine_similarity': [], 'edit_distance': [],
                   'normalized_edit_distance': [], 'jaccard': [], 'jaccard_embedding_factor': []}

        individual_results = []
        for sentence, paraphrases in sentences2paraphrases_dict.items():
            for paraphrase in paraphrases:
                result = self.evaluate_individual_sentence(sentence, paraphrase)
                individual_results.append(result)
                results['bleu_score'].append(result['bleu_score'])
                results['embedding_cosine_similarity'].append(result['embedding_cosine_similarity'])
                results['edit_distance'].append(result['edit_distance'])
                results['normalized_edit_distance'].append(result['edit_distance'])
                results['jaccard'].append(result['jaccard'])
                results['jaccard_embedding_factor'].append(result['jaccard_embedding_factor'])

        df = pd.DataFrame.from_dict(results)
        statistics = df.describe(include='all').to_dict()
        return dict(results=individual_results, statistics=statistics)


class ExtrinsicEvaluator:

    def evaluate_paraphrases(self, sentences2paraphrases_dict: Dict) -> Dict:
        deterministic(seed=42)
        input_path = os.path.join('input', 'tweets.csv')
        df = pd.read_csv(input_path)
        data = Preprocessing(CONFIGURATION['num_words'], CONFIGURATION['seq_len'], df,
                             augment=sentences2paraphrases_dict).preprocess()
        model = TextClassifier(CONFIGURATION)
        res = Run().train(model, data, CONFIGURATION)
        logging.info(res)
        return res
