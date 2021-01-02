import json
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import logging


def join_data_and_paraphrases(input_path, paraphrases, language='en'):
    df = pd.read_csv(input_path)
    df['Text'] = df['Title'] + '. ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])

    data_augmentation = {'Class Index': [], 'Text': []}
    for parafrase_cycle in paraphrases[language]:
        for i, paraphrase in enumerate(parafrase_cycle):
            data_augmentation['Text'].append(paraphrase)
            data_augmentation['Class Index'].append(df.iloc[i, 0])

    data_augmentation_df = pd.DataFrame.from_dict(data_augmentation)
    df = pd.concat([df, data_augmentation_df], axis=0, join='outer', ignore_index=True)
    return df


class Preprocessing:

    def __init__(self, num_words, seq_len, df, augment=None):
        self.data = df
        self.num_words = num_words
        self.seq_len = seq_len
        self.vocabulary = None
        self.y = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.augment = augment



    def preprocess(self):
        self.load_data()
        self.split_data()
        if self.augment:
            self.add_augmentation()
        self.clean_text()
        self.text_tokenization()
        self.build_vocabulary()
        self.word_to_idx()
        self.padding_sentences()
        # self.split_data()

        return {'x_train': self.x_train, 'y_train': self.y_train, 'x_test': self.x_test, 'y_test': self.y_test}

    def load_data(self):
        # Reads the csv file split into
        # sentences (x) and target (y)
        df = self.data
        self.x_raw = df['text'].tolist()
        self.y = df['target'].tolist()

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_raw, self.y, test_size=0.25,
                                                                                random_state=42)

    def add_augmentation(self):
        sentences_to_add = []
        labels_to_add = []
        for idx, sentence in enumerate(self.x_train):
            if sentence in self.augment:
                for paraphrase in self.augment[sentence]:
                    sentences_to_add.append(paraphrase)
                    labels_to_add.append(self.y_train[idx])
        logging.info(f'Data augmentation: added {len(sentences_to_add)} sentences')
        self.x_train.extend(sentences_to_add)
        self.y_train.extend(labels_to_add)

    def clean_text(self):
        # Removes special symbols and just keep
        # words in lower or upper form

        self.x_train = [x.lower() for x in self.x_train]
        self.x_train = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_train]

        self.x_test = [x.lower() for x in self.x_test]
        self.x_test = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_test]

    def text_tokenization(self):
        # Tokenizes each sentence by implementing the nltk tool
        self.x_train = [word_tokenize(x) for x in self.x_train]
        self.x_test = [word_tokenize(x) for x in self.x_test]

    def build_vocabulary(self):
        # Builds the vocabulary and keeps the "x" most frequent words
        self.vocabulary = dict()
        fdist = nltk.FreqDist()

        for sentence in self.x_train:
            for word in sentence:
                fdist[word] += 1

        common_words = fdist.most_common(self.num_words)

        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = (idx + 1)

    def word_to_idx(self):
        # By using the dictionary (vocabulary), it is transformed
        # each token into its index based representation

        x_train_tokenized = list()

        for sentence in self.x_train:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            x_train_tokenized.append(temp_sentence)
        self.x_train = x_train_tokenized

        x_test_tokenized = list()

        for sentence in self.x_test:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            x_test_tokenized.append(temp_sentence)
        self.x_test = x_test_tokenized

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        x_train_padded = list()

        for sentence in self.x_train:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            x_train_padded.append(sentence)

        self.x_train = np.array(x_train_padded)

        x_test_padded = list()

        for sentence in self.x_test:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            x_test_padded.append(sentence)

        self.x_test = np.array(x_test_padded)
