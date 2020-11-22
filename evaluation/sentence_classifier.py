from evaluation.model import TextClassifier
from evaluation.preprocessing import Preprocessing
from evaluation.configuration import CONFIGURATION


if __name__ == '__main__':
    pr = Preprocessing(CONFIGURATION['num_words'], CONFIGURATION['seq_len']).preprocess()

    # Initialize the model
    model = TextClassifier(CONFIGURATION)