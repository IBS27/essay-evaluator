from flask import Flask, request, render_template, url_for, jsonify
import site
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K


def sent2word(x):
    stop_words = set(stopwords.words("english"))
    x = re.sub("[^A-Za-z]", " ", x)
    x.lower()
    filtered_sentence = []
    words = x.split()
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    raw = tokenizer.tokenize(essay)
    final_words = []
    for i in raw:
        if len(i) > 0:
            final_words.append(sent2word(i))
    return final_words


def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.0
    index2word_set = set(model.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[i])
    vec = np.divide(vec, noOfWords)
    return vec


def getVecs(essays, model, num_features):
    c = 0
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c += 1
    return essay_vecs


def get_model():
    model = Sequential()
    model.add(
        LSTM(
            300,
            dropout=0.4,
            recurrent_dropout=0.4,
            input_shape=[1, 300],
            return_sequences=True,
        )
    )
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="relu"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["mae"])
    model.summary()
    return model


def convertToVec(text, essay_num):
    content = text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format(
            f"model/essay{essay_num}/word2vecmodel.bin", binary=True
        )
        clean_test_essays = []
        clean_test_essays.append(sent2word(content))
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(
            testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1])
        )

        lstm_model = load_model(f"model/essay{essay_num}/final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def submit():
    essay1 = request.form["essay1"]
    score1 = convertToVec(essay1, 1)

    essay2 = request.form["essay2"]
    score2 = convertToVec(essay2, 2)

    essay3 = request.form["essay3"]
    score3 = convertToVec(essay3, 3)

    essay4 = request.form["essay4"]
    score4 = convertToVec(essay4, 4)

    essay5 = request.form["essay5"]
    score5 = convertToVec(essay5, 5)

    scores = [score1, score2, score3, score4, score5]
    final_scores = []
    for score in scores:
        if score is None:
            continue
        else:
            final_scores.append(int(score))

    total_score = f"Score: {sum(final_scores) * 2}/100"
    return render_template("index.html", total_score=total_score)


if __name__ == "__main__":
    app.run()
