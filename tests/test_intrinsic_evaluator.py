import nltk
import pylev
from numpy import dot
from numpy.linalg import norm
from typing import List, Dict
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def evaluate_individual_sentence(original_sentence: str, paraphrase: str) -> Dict:
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

sentences2paraphrases_dict = {"I'm 14 years old.": ["I am 14 years old.", "I'm 14."]}

results = {'bleu_score': [], 'cosine_similarity': [], 'edit_distance': [], 'cos_edit_distance': [],
                   'original_sentence': [], 'paraphrase': []}

for sentence, paraphrases in sentences2paraphrases_dict.items():
    for paraphrase in paraphrases:
        result = evaluate_individual_sentence(sentence, paraphrase)
        results['bleu_score'].append(result['bleu_score'])
        results['cosine_similarity'].append(result['cosine_similarity'])
        results['edit_distance'].append(result['edit_distance'])
        results['cos_edit_distance'].append(result['cos_edit_distance'])
        results['original_sentence'].append(sentence)
        results['paraphrase'].append(paraphrases)

df = pd.DataFrame.from_dict(results)
statistics = df.describe(include='all').to_dict()
print(statistics)