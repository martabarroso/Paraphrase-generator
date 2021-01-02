import pandas as pd
import time
import os
import git
import uuid
import logging
from paraphraser.utils import init_logging, deterministic

from extrinsic_evaluation.model import TextClassifier
from extrinsic_evaluation.preprocessing import Preprocessing
from extrinsic_evaluation.configuration import CONFIGURATION
from extrinsic_evaluation.run import Run


if __name__ == '__main__':
    deterministic(seed=42)
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    output_path = 'output'
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    output_dir = os.path.join(output_path,
                              f'baseline-{timestamp}-{sha[:4]}-{extra_id[:4]}')

    os.makedirs(output_dir)
    init_logging(os.path.join(output_dir, 'baseline.log'))

    input_path = os.path.join('input', 'tweets.csv')
    df = pd.read_csv(input_path)
    data = Preprocessing(CONFIGURATION['num_words'], CONFIGURATION['seq_len'], df,
                         augment=None).preprocess()
    model = TextClassifier(CONFIGURATION)
    res = Run().train(model, data, CONFIGURATION)
    logging.info(res)
