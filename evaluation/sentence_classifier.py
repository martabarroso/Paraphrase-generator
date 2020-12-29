from evaluation.model import TextClassifier
from evaluation.preprocessing import Preprocessing, join_data_and_paraphrases
from evaluation.configuration import CONFIGURATION
from evaluation.run import Run

import pandas as pd


if __name__ == '__main__':
    paraphrases = None
    '''
    paraphrases = {'en': [['When they go on field trips, they let parents go and help.',
                           'Here we can only take one bus.'],
                          ['When they go on excursions, they let their parents go and help.',
                           'Here we can only take a bus.']],
                   'de': [['Wenn sie auf Exkursionen gehen, lassen sie die Eltern gehen und helfen.',
                           'Hier können wir nur einen Bus nehmen.']]}
    language = 'en'
    '''

    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')
    df = pd.concat([df_train, df_test], axis=0, join='outer')
    df = join_data_and_paraphrases(df, paraphrases, language='en')


    # Initialize the model
    model = TextClassifier(CONFIGURATION)

    # Preprocessing
    preprocessing = Preprocessing(CONFIGURATION['num_words'], CONFIGURATION['seq_len'], df)
    data = preprocessing.preprocess()

    # Training and evaluation
    Run().train(model, data, CONFIGURATION)

