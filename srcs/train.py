import sys
import spacy
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
sys.path.append('srcs')
import utils

SPACY_MODEL = 'en_core_web_md'
EMBED_DIM = 300
EPOCHS = 10
BATCH_SIZE = 128
MAX_LEN = 200
LABEL_DICT = {
    'positive': 0,
    'negative': 1,
}


def main():
    """ """
    # load IMDB review data
    df = pd.read_csv('data/IMDB Dataset.csv')
    # split train test
    df_train = df.sample(frac=0.8, random_state=123)
    df_test = df.drop(df_train.index)

    # load spacy model
    nlp = spacy.load(SPACY_MODEL, disable=['tagger', 'parser', 'ner'])

    # encode data
    train_x, train_y = [], []
    test_x, test_y = [], []
    print('encoding training data...')
    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        train_x.append(utils.encode_text(row.review, nlp, MAX_LEN, EMBED_DIM))
        train_y.append(utils.encode_label(row.sentiment, LABEL_DICT))

    print('encoding validation data...')
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        test_x.append(utils.encode_text(row.review, nlp, MAX_LEN, EMBED_DIM))
        test_y.append(utils.encode_label(row.sentiment, LABEL_DICT))

    print('convert into arrays')
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # construct model
    input_embed = tf.keras.layers.Input(shape=(MAX_LEN, EMBED_DIM), name='input_embedding')
    hidden = tf.keras.layers.BatchNormalization()(input_embed)
    hidden = tf.keras.layers.LSTM(100, return_sequences=True)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.SpatialDropout1D(0.2)(hidden)
    hidden = tf.keras.layers.MultiHeadAttention(30, 6)(hidden, hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.GlobalAveragePooling1D()(hidden)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    dense = tf.keras.layers.Dense(24, activation='relu')(hidden)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.Dropout(0.1)(dense)
    output = tf.keras.layers.Dense(2, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=input_embed, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # train model
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=(test_x, test_y),
              batch_size=BATCH_SIZE)

    # save model
    model.save('model')


if __name__ == '__main__':
    main()
