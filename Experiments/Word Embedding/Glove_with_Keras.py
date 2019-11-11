from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

# For cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100 # only can be 50, 100, 200, or 300
VALIDATION_SPLIT = 0.2


# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}

with open(os.path.join(GLOVE_DIR, 'glove.6B.{}d.txt'.format(EMBEDDING_DIM))) as f:
    # The file looks like this
    # the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 ...
    # of -0.1529 -0.24279 0.89837 0.16996 0.53516 0.48784 -0.58826 ...
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        # "coefs" is an array that represents the word vector in 100 dimensions [-0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 ... ]
        embeddings_index[word] = coefs
        # "embeddings_index" looks like this:
        # {
        #   the: [-0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 ... ]
        #   of: [-0.1529 -0.24279 0.89837 0.16996 0.53516 0.48784 -0.58826 ...]
        #   ...
        # }

print('Found %s word vectors.' % len(embeddings_index))

predict_num_days_ahead = 2
# second, prepare text samples and their labels
# Import CSV
df_raw_1 = pd.read_csv('./data/GOOGL_data_with_news_3.csv', index_col="date", parse_dates=True)
df_raw_1['tagged_news'] = df_raw_1['tagged_news'].replace(np.nan, '', regex=True)
# df_raw_1 = df_raw_1.resample('B').pad()
# df_raw_1.index.freq = 'B'

df_with_targets = df_raw_1.copy()
df_with_targets['target_reg'] = df_with_targets['open'].shift(-predict_num_days_ahead)
df_with_targets = df_with_targets.iloc[:-1]
df_with_targets['target_class'] = df_with_targets['target_reg'] > df_with_targets['close']
df_1 = df_with_targets.copy()
df_1 = df_1.dropna()

df_raw_2 = pd.read_csv('./data/Facebook_with_news.csv', index_col="date", parse_dates=True)
df_raw_2['tagged_news'] = df_raw_2['tagged_news'].replace(np.nan, '', regex=True)
# df_raw_2 = df_raw_2.resample('B').pad()
# df_raw_2.index.freq = 'B'

df_with_targets_2 = df_raw_2.copy()
df_with_targets_2['target_reg'] = df_with_targets_2['open'].shift(-predict_num_days_ahead)
df_with_targets_2 = df_with_targets_2.iloc[:-1]
df_with_targets_2['target_class'] = df_with_targets_2['target_reg'] > df_with_targets_2['close']
df_2 = df_with_targets_2.copy()
df_2 = df_2.dropna()

df_raw_3 = pd.read_csv('./data/Microsoft_with_news.csv', index_col="date", parse_dates=True)
df_raw_3['tagged_news'] = df_raw_3['tagged_news'].replace(np.nan, '', regex=True)
# df_raw_3 = df_raw_3.resample('B').pad()
# df_raw_3.index.freq = 'B'

df_with_targets_3 = df_raw_3.copy()
df_with_targets_3['target_reg'] = df_with_targets_3['open'].shift(-predict_num_days_ahead)
df_with_targets_3 = df_with_targets_3.iloc[:-1]
df_with_targets_3['target_class'] = df_with_targets_3['target_reg'] > df_with_targets_3['close']
df_3 = df_with_targets_3.copy()
df_3 = df_3.dropna()


# texts = df['main_news'].values
texts = np.concatenate((df_1['main_news'].values, df_2['main_news'].values, df_3['main_news'].values), axis=None)
# labels = df['target_class'].values
labels = np.concatenate((df_1['target_class'].values, df_2['target_class'].values, df_3['target_class'].values), axis=None)

print(labels)
# "texts" is bascially a long array of texts - Eg. ["Hello there.", "Hello, how are you? There is food"]
# "labels" is bascially a long array of target labels - eg. [0, 1, 3, 1, 2, 0, 1]
print(texts)
print('Found %s texts.' % len(texts))
print('Found %s labels.' % len(labels))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# Converts all the texts to an array of numbers
# Eg. ["Hello there.", "Hello, how are you? There is food"]
# Convert to: [[1, 2], [1, 3, 4, 5, 2, 6, 7]]

tokenized_word_index_map = tokenizer.word_index
# eg. {'hello': 1, 'there': 2, 'how': 3, 'are': 4, 'you': 5, 'is': 6, 'food': 7}
print(tokenized_word_index_map)
print('Found %s unique tokens.' % len(tokenized_word_index_map))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# Pads the sequences
# Eg. [[1, 2], [1, 3, 4, 5, 2, 6, 7]]
# Convert to: [[0, 0, 0, 0, 0, 0, 0, 1, 2], [0, 0, 1, 3, 4, 5, 2, 6, 7]]

labels = to_categorical(np.asarray(labels))
# One-hot encoding for labels
# Eg. [0, 1, 3, 1, 2, 0, 1]
# Convert to: 
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 0. 1.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [1. 0. 0. 0.]
#  [0. 1. 0. 0.]]
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(tokenized_word_index_map) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in tokenized_word_index_map.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector




'''
print('Training model.')

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences) # note how this is a 1D convolution instead of a 2D convolution
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)

preds = Dense(2, activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=30,
          validation_data=(x_val, y_val))

predictions = model.predict(x_val)
target = y_val
'''


def create_network():
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(256, 4, activation='relu')(embedded_sequences) # 128 is the number of filters, 4 is the kernel size (in this case, a 4-gram)
    x = MaxPooling1D(3)(x) # 3 is the size of the pooling window
    x = Conv1D(128, 4, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 4, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)

    preds = Dense(2, activation='softmax')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    return model

neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=50, 
                                 batch_size=64, 
                                 verbose=0)

score = cross_val_score(neural_network, data, labels, cv=8)
print(score)
sum = 0
for i in range(len(score)):
    sum += score[i]
print(sum/len(score))





