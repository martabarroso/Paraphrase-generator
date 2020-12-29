import pandas as pd
from io import StringIO
from unittest import TestCase
import numpy as np
from pandas._testing import assert_frame_equal

from evaluation.preprocessing import Preprocessing, join_data_and_paraphrases


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        # num_words: selection of most common num_words
        df = pd.read_csv('../resources/train.csv')
        self.preprocessing = Preprocessing(num_words=10, seq_len=10, df=df)

    def test_join_data_and_paraphrases(self):
        expected_data_aug_df = '''Class Index,Text
1,"Title 1. When they go on field trips, they let parents go and help."
2,Title 2. Here we can only take one bus.
1,"When they go on field trips, they let parents go and help."
2,Here we can only take one bus.
1,"When they go on excursions, they let their parents go and help."
2,Here we can only take a bus.'''
        expected_data_aug_df = pd.read_csv(StringIO(expected_data_aug_df))

        input_path = '../resources/train.csv'
        paraphrases = {'en': [['When they go on field trips, they let parents go and help.',
                            'Here we can only take one bus.'],
                           ['When they go on excursions, they let their parents go and help.',
                            'Here we can only take a bus.']],
                       'de': [['Wenn sie auf Exkursionen gehen, lassen sie die Eltern gehen und helfen.',
                            'Hier können wir nur einen Bus nehmen.']]}
        language = 'en'
        df = join_data_and_paraphrases(input_path, paraphrases, language)
        assert_frame_equal(expected_data_aug_df, df)

    def test_load_data(self):
        expected_x_raw = ['Title 1. When they go on field trips, they let parents go and help.',
                          'Title 2. Here we can only take one bus.',
                          'When they go on field trips, they let parents go and help.',
                          'Here we can only take one bus.',
                          'When they go on excursions, they let their parents go and help.',
                          'Here we can only take a bus.']
        expected_y = [1, 2, 1, 2, 1, 2]

        data_aug_df = '''Class Index,Text
        1,"Title 1. When they go on field trips, they let parents go and help."
        2,Title 2. Here we can only take one bus.
        1,"When they go on field trips, they let parents go and help."
        2,Here we can only take one bus.
        1,"When they go on excursions, they let their parents go and help."
        2,Here we can only take a bus.'''
        data_aug_df = pd.read_csv(StringIO(data_aug_df))

        self.preprocessing.data = data_aug_df
        self.preprocessing.load_data()
        self.assertListEqual(expected_x_raw, self.preprocessing.x_raw)
        self.assertListEqual(expected_y, self.preprocessing.y)

    def test_clean_text(self):
        # no punctuation, no numbers
        expected_x_raw = ['title when they go on field trips they let parents go and help ',
                          'title here we can only take one bus ',
                          'when they go on field trips they let parents go and help ',
                          'here we can only take one bus ',
                          'when they go on excursions they let their parents go and help ',
                          'here we can only take a bus ']

        self.preprocessing.x_raw = ['Title 1. When they go on field trips, they let parents go and help.',
                                    'Title 2. Here we can only take one bus.',
                                    'When they go on field trips, they let parents go and help.',
                                    'Here we can only take one bus.',
                                    'When they go on excursions, they let their parents go and help.',
                                    'Here we can only take a bus.']
        self.preprocessing.clean_text()
        self.assertListEqual(expected_x_raw, self.preprocessing.x_raw)

    def test_tokenization(self):
        import nltk
        import ssl

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt')
        expected_tokens = [['title', 'when', 'they', 'go', 'on', 'field', 'trips', 'they', 'let', 'parents', 'go',
                            'and', 'help'], ['title', 'here', 'we', 'can', 'only', 'take', 'one', 'bus'],
                           ['when', 'they', 'go', 'on', 'field', 'trips', 'they', 'let', 'parents', 'go', 'and',
                            'help'], ['here', 'we', 'can', 'only', 'take', 'one', 'bus'], ['when', 'they', 'go', 'on',
                                                                                           'excursions', 'they', 'let',
                                                                                           'their', 'parents', 'go',
                                                                                           'and', 'help'],
                           ['here', 'we', 'can', 'only', 'take', 'a', 'bus']]

        self.preprocessing.x_raw = ['title when they go on field trips they let parents go and help ',
                                    'title here we can only take one bus ',
                                    'when they go on field trips they let parents go and help ',
                                    'here we can only take one bus ',
                                    'when they go on excursions they let their parents go and help ',
                                    'here we can only take a bus ']
        self.preprocessing.text_tokenization()
        self.assertListEqual(expected_tokens, self.preprocessing.x_raw)

    def test_build_vocabulary(self):
        expected_vocabulary = {'they': 1, 'go': 2, 'when': 3, 'on': 4, 'let': 5, 'parents': 6, 'and': 7, 'help': 8,
                               'here': 9, 'we': 10}

        self.preprocessing.x_raw = [['title', 'when', 'they', 'go', 'on', 'field', 'trips', 'they', 'let', 'parents',
                                     'go', 'and', 'help'], ['title', 'here', 'we', 'can', 'only', 'take', 'one', 'bus'],
                                    ['when', 'they', 'go', 'on', 'field', 'trips', 'they', 'let', 'parents', 'go',
                                     'and', 'help'], ['here', 'we', 'can', 'only', 'take', 'one', 'bus'],
                                    ['when', 'they', 'go', 'on', 'excursions', 'they', 'let', 'their', 'parents', 'go',
                                     'and', 'help'], ['here', 'we', 'can', 'only', 'take', 'a', 'bus']]
        self.preprocessing.build_vocabulary()
        self.assertDictEqual(expected_vocabulary, self.preprocessing.vocabulary)

    def test_word_to_idx(self):
        expected_x_tokenized = [[3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10], [3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10],
                                [3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10]]
        self.preprocessing.vocabulary = {'they': 1, 'go': 2, 'when': 3, 'on': 4, 'let': 5, 'parents': 6, 'and': 7,
                                         'help': 8, 'here': 9, 'we': 10}
        self.preprocessing.x_raw = [['title', 'when', 'they', 'go', 'on', 'field', 'trips', 'they', 'let', 'parents',
                                     'go','and', 'help'], ['title', 'here', 'we', 'can', 'only', 'take', 'one', 'bus'],
                                    ['when', 'they', 'go', 'on', 'field', 'trips', 'they', 'let', 'parents', 'go',
                                     'and', 'help'], ['here', 'we', 'can', 'only', 'take', 'one', 'bus'],
                                    ['when', 'they', 'go', 'on', 'excursions', 'they', 'let', 'their', 'parents', 'go',
                                     'and', 'help'], ['here', 'we', 'can', 'only', 'take', 'a', 'bus']]
        self.preprocessing.word_to_idx()
        self.assertListEqual(expected_x_tokenized, self.preprocessing.x_tokenized)

    def test_padding_sentences(self):
        expected_x_padded = np.array([[3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.preprocessing.x_tokenized = [[3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10], [3, 1, 2, 4, 1, 5, 6, 2, 7, 8],
                                          [9, 10], [3, 1, 2, 4, 1, 5, 6, 2, 7, 8], [9, 10]]
        self.preprocessing.padding_sentences()
        np.array_equal(expected_x_padded, self.preprocessing.x_padded)

