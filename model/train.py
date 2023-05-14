# Import necessary libraries and modules
import pandas as pd
import numpy as np
from pre_process import essay_to_sentence, get_avg_feature_vecs
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from gensim.models import Word2Vec

# Load essay dataset
data = pd.read_csv(
    '~/Developer/essay-evaluator/model/training_dataset.tsv',
    sep='\t'
)
essays = data['essay'].values.tolist()
scores = data['essay_score'].values.tolist()

# Set the Word2Vec parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Train the Word2Vec model
model = Word2Vec(essays, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context, sample=downsampling)

# Preprocess the essays
max_length = max([len(s.split()) for s in essays])
essays = essay_to_sentence(essays)
X = get_avg_feature_vecs(essays, model, num_features)
X = pad_sequences(X, maxlen=max_length, padding='post', truncating='post')

# Define the 2-layer LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, num_features), return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the number of folds for cross-validation
num_folds = 5

# Initialize the list of kappa scores for each fold
kappas = []

# Use KFold cross-validation to evaluate the model
kf = KFold(n_splits=num_folds, shuffle=True)
for train_index, test_index in kf.split(X):
    train_index = np.array(train_index)
    test_index = np.array(test_index)
    X_train, y_train = X[train_index], scores[train_index]
    X_test, y_test = X[test_index], scores[test_index]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 300))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 300))
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
    # Split the data into training and testing sets for this fold
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = scores[train_index], scores[test_index]
    # X_train, y_train = X[train_index.astype(int)], scores[train_index.astype(int)]
    # X_val, y_val = X[val_index.astype(int)], scores[val_index.astype(int)]

    # Set the maximum sequence length
    # max_seq_length = 1064
    # Truncate X_train so that it has a size that is a multiple of 10380
    # X_train = X_train[:-(X_train.shape[0] % 10380)]

    # Reshape X_train to have shape (n_samples, 1064, 300)
    # X_train = np.reshape(X_train, (-1, 1064, 300))

    # Pad or truncate the sequences to a fixed length
    # X_train = pad_sequences(X_train, maxlen=max_seq_length)
    # X_test = pad_sequences(X_test, maxlen=max_seq_length)

    # Reshape the input data
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 300))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 300))

    # Convert the scores to categorical labels
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # Train the model on the training set for this fold
    # model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

    # Evaluate the model on the testing set for this fold
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate the kappa score for this fold and append it to the list
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    kappas.append(kappa)

# Print the average kappa score after the cross-validation
print('Average Kappa:', np.mean(kappas))

# Save the model for later use
# model.save('essay_evaluator.h5')
