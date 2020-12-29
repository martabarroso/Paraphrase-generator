import json
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


def join_data_and_paraphrases(input_path, paraphrases, language='en'):
    df = pd.read_csv(input_path)
    df['Text'] = df['Title'] + '. ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])

    data_augmentation = {'Class Index':[], 'Text':[]}
    for parafrase_cycle in paraphrases[language]:
        for i, paraphrase in enumerate(parafrase_cycle):
            data_augmentation['Text'].append(paraphrase)
            data_augmentation['Class Index'].append(df.iloc[i, 0])

    data_augmentation_df = pd.DataFrame.from_dict(data_augmentation)
    df = pd.concat([df, data_augmentation_df], axis=0, join='outer', ignore_index=True)
    return df


class Preprocessing:

    def __init__(self, num_words, seq_len, df):
        self.data = df
        self.num_words = num_words
        self.seq_len = seq_len
        self.vocabulary = None
        self.x_tokenized = None
        self.x_padded = None
        self.x_raw = None
        self.y = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        self.load_data()
        self.clean_text()
        self.text_tokenization()
        self.build_vocabulary()
        self.word_to_idx()
        self.padding_sentences()
        self.split_data()

        return {'x_train': self.x_train, 'y_train': self.y_train, 'x_test': self.x_test, 'y_test': self.y_test}

    def load_data(self):
        # Reads the csv file split into
        # sentences (x) and target (y)
        df = self.data
        self.x_raw = df['Text'].tolist()
        self.y = df['Class Index'].tolist()

    def clean_text(self):
        # Removes special symbols and just keep
        # words in lower or upper form

        self.x_raw = [x.lower() for x in self.x_raw]
        self.x_raw = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_raw]

    def text_tokenization(self):
        # Tokenizes each sentence by implementing the nltk tool
        self.x_raw = [word_tokenize(x) for x in self.x_raw]

    def build_vocabulary(self):
        # Builds the vocabulary and keeps the "x" most frequent words
        self.vocabulary = dict()
        fdist = nltk.FreqDist()

        for sentence in self.x_raw:
            for word in sentence:
                fdist[word] += 1

        common_words = fdist.most_common(self.num_words)

        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = (idx + 1)

    def word_to_idx(self):
        # By using the dictionary (vocabulary), it is transformed
        # each token into its index based representation

        self.x_tokenized = list()

        for sentence in self.x_raw:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            self.x_tokenized.append(temp_sentence)

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        self.x_padded = list()

        for sentence in self.x_tokenized:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            self.x_padded.append(sentence)

        self.x_padded = np.array(self.x_padded)

    # Since we have train and test files we do not need this
    # TODO repeat the process for train and test files!
    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_padded, self.y, test_size=0.25,
                                                                                random_state=42)