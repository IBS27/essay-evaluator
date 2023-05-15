# Import necessary libraries and modules
import pandas as pd
import tensorflow as tf

# import numpy as np
from numpy import *
from pre_process import Tokenizer, essay_to_sentence, get_avg_feature_vecs
from keras_preprocessing.sequence import pad_sequences

# from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import KFold

# from sklearn.metrics import cohen_kappa_score
from gensim.models import Word2Vec


# Set the Word2Vec parameters
num_features = 100  # Word vector dimensionality
# num_features = 1  # Word vector dimensionality
min_word_count = 1  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 5  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words


def create_model(max_length, num_features):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_length, num_features), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="relu"))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model


# Load essay dataset
print("Loading essays")
data = pd.read_csv("~/Developer/essay-evaluator/model/training_dataset_2.tsv", sep="\t")
essays = data["essay"].apply(lambda x: essay_to_sentence(x)).tolist()
essays = [str(essay) for essay in essays]
scores = data["essay_score"]
print("Loaded essays")
# scores = data["essay_score"].values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(essays)
vocab_size = len(tokenizer.word_index) + 1
max_length = 100

# model = Word2Vec.load("model.bin")
model = Word2Vec(
    essays,
    workers=num_workers,
    vector_size=num_features,
    min_count=min_word_count,
    window=context,
    # sample=downsampling,
)
num_features = model.vector_size

# Train the Word2Vec model
# model = Word2Vec(
#     essays,
#     workers=num_workers,
#     vector_size=num_features,
#     min_count=min_word_count,
#     window=context,
#     sample=downsampling,
# )

# Preprocess the essays
# max_length = max([len(s.split()) for s in essays])
# essays = essay_to_sentence(essays)
# X = get_avg_feature_vecs(essays, model, num_features)
# X = pad_sequences(X, maxlen=max_length, padding="post", truncating="post")

# Define the 2-layer LSTM model
# model = Sequential()
# # model.add(LSTM(128, input_shape=(1064, 1), return_sequences=True))
# model.add(LSTM(128, input_shape=(max_length, num_features), return_sequences=True))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dense(5, activation="softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define the number of folds for cross-validation
num_folds = 5

# Initialize the list of kappa scores for each fold
kappas = []

# # Use KFold cross-validation to evaluate the model
kf = KFold(n_splits=num_folds, shuffle=True)
# for train_index, test_index in kf.split(X):
#     train_index = np.array(train_index)
#     test_index = np.array(test_index)
#     X_train, y_train = X[train_index.astype(int)], scores[train_index.astype(int)]
#     X_test, y_test = X[test_index.astype(int)], scores[test_index.astype(int)]
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     # Train the model on the training set for this fold
#     model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

#     # Evaluate the model on the testing set for this fold
#     y_pred = model.predict(X_test)
#     y_pred = np.argmax(y_pred, axis=1)
#     y_true = np.argmax(y_test, axis=1)


#     # Calculate the kappa score for this fold and append it to the list
#     kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
#     kappas.append(kappa)
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


# for train_indices, val_indices in kf.split(essays):
#     print("essays")
#     print(essays)
#     print("train")
#     print(train_indices)
#     print("val")
#     print(val_indices)
#     print("cross validating")

#     essays = flatten_list(essays)
#     essays = flatten_list(essays)
#     print("reshape")
#     print(essays)
#     # Split the data into training and validation sets
#     train_essays = essays[train_indices]
#     val_essays = essays[val_indices]
#     train_labels = data["score"][train_indices].values
#     val_labels = data["score"][val_indices].values

#     # Preprocess the training and validation essays
#     X_train = get_avg_feature_vecs(train_essays, model, num_features)
#     X_val = get_avg_feature_vecs(val_essays, model, num_features)

#     # Pad the feature vectors to a maximum length of 100
#     X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
#     X_val = pad_sequences(X_val, maxlen=max_length, padding="post")

#     # Define the model
#     model = create_model(max_length, num_features)

#     # Train the model
#     model.fit(X_train, train_labels, epochs=50, batch_size=64)

#     # Make predictions on the validation set
#     y_pred = model.predict(X_val)

#     # Calculate the Quadratic Weighted Kappa
#     kappa = cohen_kappa_score(val_labels.round(), y_pred.round(), weights="quadratic")
#     kappas.append(kappa)


# Print the average kappa score after the cross-validation
# print("Average Kappa:", np.mean(kappas))

# # Save the model for later use
# # model.save('essay_evaluator.h5')
for i, (train_index, test_index) in enumerate(kf.split(essays)):
    train_index = [int(x) for x in train_index]
    test_index = [int(x) for x in test_index]
    print(train_index)
    print(test_index)
    x_train = [essays[x] for x in train_index]
    y_train = [scores[x] for x in train_index]
    # x_train, y_train = essays[train_index], scores[train_index]
    x_test = [essays[x] for x in test_index]
    y_test = [scores[x] for x in test_index]
    # x_test, y_test = essays[test_index], scores[test_index]

    # Preprocess the training and validation essays
    X_train = get_avg_feature_vecs(x_train, model, num_features)
    X_test = get_avg_feature_vecs(x_test, model, num_features)

    # Pad the feature vectors to a maximum length of 100
    X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
    X_test = pad_sequences(X_test, maxlen=max_length, padding="post")
    data_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    X_train = X_train.reshape(-1, max_length, num_features)
    X_test = X_test.reshape(-1, max_length, num_features)
    # y_train = y_train.reshape(-1, max_length, num_features)
    # y_test = y_test.reshape(-1, max_length, num_features)

    model = create_model(max_length, num_features)
    # model.fit(X_train, y_train, epochs=50, batch_size=64)
    model.fit(data_set)
    score = model.evaluate(x_test, y_test)

    print("Fold:", i + 1, "Test score:", score)
