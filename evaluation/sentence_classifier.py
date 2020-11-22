from evaluation.model import TextClassifier
from evaluation.preprocessing import Preprocessing
from evaluation.configuration import CONFIGURATION
from evaluation.run import Run


if __name__ == '__main__':
    data = Preprocessing(CONFIGURATION['num_words'], CONFIGURATION['seq_len'], './input.csv').preprocess()

    # Initialize the model
    model = TextClassifier(CONFIGURATION)

    Run().train(model, data, CONFIGURATION)
