from math import isnan
from flask import Flask, request, render_template
from gensim.models import word2vec
import numpy as np
from preprocess import *
import nltk
import re
from nltk.corpus import stopwords
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
import keras.backend as K
from gensim.models.keyedvectors import KeyedVectors


def get_model():
    """Define the model."""
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


def get_preds(content=""):
    num_features = 300
    model = word2vec.Word2VecKeyedVectors.load_word2vec_format(
        "./model/word2vecmodel.bin", binary=True
    )
    clean_test_essays = []
    clean_test_essays.append(essay_to_wordlist(content, remove_stopwords=True))
    test_data_vecs = getAvgFeatureVecs(clean_test_essays, model, num_features)
    test_data_vecs = np.array(test_data_vecs)
    test_data_vecs = np.reshape(
        test_data_vecs, (test_data_vecs.shape[0], 1, test_data_vecs.shape[1])
    )

    lstm_model = get_model()
    lstm_model.load_weights("./model/final_lstm.h5")
    preds = lstm_model.predict(test_data_vecs)

    if len(content) < 1000:
        preds = 0

    return preds


def get_score(preds, max_score):
    if isnan(preds):
        preds = 0
    else:
        preds = np.around(preds)

    if preds < 0 or preds >= max_score or (preds / max_score) <= 0.5:
        preds = 0

    score = np.around((preds / max_score) * 10) * 2
    return int(score)


maximum_scores = [12, 6, 5, 30, 60]

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def submit():
    essay1 = request.form["essay1"]
    preds1 = get_preds(essay1)
    score1 = get_score(preds1, maximum_scores[0])

    essay2 = request.form["essay2"]
    preds2 = get_preds(essay2)
    score2 = get_score(preds2, maximum_scores[1])

    essay3 = request.form["essay3"]
    preds3 = get_preds(essay3)
    score3 = get_score(preds3, maximum_scores[2])

    essay4 = request.form["essay4"]
    preds4 = get_preds(essay4)
    score4 = get_score(preds4, maximum_scores[3])

    essay5 = request.form["essay5"]
    preds5 = get_preds(essay5)
    score5 = get_score(preds5, maximum_scores[4])

    scores = [score1, score2, score3, score4, score5]
    scores = np.int_(scores)

    K.clear_session()

    total_score = f"Score: {sum(scores)}/100"
    return render_template("index.html", total_score=total_score)


if __name__ == "__main__":
    app.run()
