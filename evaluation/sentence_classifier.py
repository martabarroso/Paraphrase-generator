from evaluation.model import TextClassifier
from evaluation.preprocessing import Preprocessing, join_data_and_paraphrases
from evaluation.configuration import CONFIGURATION
from evaluation.run import Run


if __name__ == '__main__':
    input_path = '../resources/train.csv'
    paraphrases = {'en': [['When they go on field trips, they let parents go and help.',
                           'Here we can only take one bus.'],
                          ['When they go on excursions, they let their parents go and help.',
                           'Here we can only take a bus.']],
                   'de': [['Wenn sie auf Exkursionen gehen, lassen sie die Eltern gehen und helfen.',
                           'Hier können wir nur einen Bus nehmen.']]}
    language = 'en'

    # Perform data augmentation
    df = join_data_and_paraphrases(input_path, paraphrases, language='en')

    # Initialize the model
    model = TextClassifier(CONFIGURATION)

    # Preprocessing
    preprocessing = Preprocessing(CONFIGURATION['num_words'], CONFIGURATION['seq_len'], df)
    data = preprocessing.preprocess()

    # Training and evaluation
    Run().train(model, data, CONFIGURATION)

