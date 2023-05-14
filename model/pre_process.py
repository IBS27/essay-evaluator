# Import required libraries
import re
import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer


def essay_to_wordlist(essay, remove_stopwords=True):
    essay = re.sub(r'<.*?', '', str(essay))
    words = essay.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [word for word in words if word not in stops]

    return ' '.join(words)


def essay_to_sentence(essays):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    for essay in essays:
        raw_sentences = tokenizer.tokenize(essay.strip())
        sentences.append([essay_to_wordlist(s) for s in raw_sentences if s])
    return sentences


def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype='float32')

    n_words = 0.

    for word in words:
        if word in model.wv.index_to_key:
            feature_vec = np.add(feature_vec, model.wv[word])
            n_words += 1.

    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)

    return feature_vec


def get_avg_feature_vecs(essays, model, num_features):
    essay_feature_vecs = np.zeros((len(essays), num_features), dtype='float32')

    for i, essay in enumerate(essays):
        words = essay_to_wordlist(essay)

        feature_vecs = make_feature_vec(words, model, num_features)

        avg_feature_vec = np.divide(
            np.sum(feature_vecs, axis=0), len(feature_vecs)
        )

        essay_feature_vecs[i] = avg_feature_vec

    return essay_feature_vecs
