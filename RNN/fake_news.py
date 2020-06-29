from os import path
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import numpy as np

# Loading Data

BASE_DIR = path.dirname(__file__)
CSV_PATH = path.join(BASE_DIR, "../datasets/fake.csv")
data = pd.read_csv(CSV_PATH, sep=",", header=0)

FEATURES = ["title", "text"]
LABELS = ["type"]
CLASS_NAMES = [
    'bias', 'conspiracy', 'fake', 'bs', 'satire', 'hate', 'junksci', 'state']

# One-hot Encoding of the Labels

y_data = []

for class_name in list(data["type"]):
    bag = [0 for _ in range(len(CLASS_NAMES))]

    for i, label in enumerate(CLASS_NAMES):
        if class_name == label:
            bag[i] = 1
            break

    y_data.append(bag)

y_data = np.array(y_data, dtype=np.int32)

# Preprocessing input data

data_title = list(data['title'])
data_text = list(data['text'])


def ConvertElementsFromFloatToString(texts_list):
    for i, x in enumerate(texts_list):
        if type(x) == float:
            texts_list[i] = str(x)


ConvertElementsFromFloatToString(data_title)
ConvertElementsFromFloatToString(data_text)


'''
idx = 4
print(len(data_title[idx]), len(data_title))
print()
print(len(data_text[idx]), len(data_text))
'''

# Tokenizing

num_words_title = 5000
num_words_text = 10000


class TokenizerWrap(Tokenizer):
    def __init__(self, texts, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.tokens = self.texts_to_sequences(texts)
        self.tokens_length = [len(x) for x in self.tokens]
        self.max_tokens = np.mean(self.tokens_length) + \
            2 * np.std(self.tokens_length)
        self.max_tokens = int(self.max_tokens)

        self.tokens_padded = pad_sequences(
            self.tokens, maxlen=self.max_tokens, truncating='post')

    def tokens_to_string(self, tokens):
        words = [self.index_word[t] for t in tokens if t != 0]
        string = " ".join(words)

        return string

    def string_to_tokens(self, string):
        self.fit_on_texts([string])
        tokens = self.texts_to_sequences([string])
        tokens = pad_sequences(
            tokens, maxlen=self.num_words, truncating='post')

        return tokens


tokenizer_title = TokenizerWrap(data_title, num_words=num_words_title)
tokens_title = tokenizer_title.tokens_padded

tokenizer_text = TokenizerWrap(data_text, num_words=num_words_text)
tokens_text = tokenizer_text.tokens_padded

x_data_title = tokens_title
x_data_text = tokens_text
'''
# Building Model

embedding_dims = 256
rnn_units = 512
dense_units = len(CLASS_NAMES)


def build_model(embedding_dims, rnn_units, dense_units):
    classifier = Sequential([
        Embedding(input_dim=num_words_text, output_dim=embedding_dims,
                  batch_input_shape=(None, None)),
        LSTM(rnn_units, return_sequences=True),
        LSTM(rnn_units, return_sequences=True),
        LSTM(rnn_units, return_sequences=False),
        Dense(dense_units, activation='softmax')
    ])

    return classifier


classifier_title = build_model(embedding_dims, rnn_units, dense_units)
classifier_text = build_model(embedding_dims, rnn_units, dense_units)

# Model Summary

classifier_title.summary()
classifier_text.summary()

# Compiling Model

classifier_title.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier_text.compile(
    optimizer=RMSprop(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


path_checkpoint = './checkpoint_dir/'
callback_checkpoint = ModelCheckpoint(
    path_checkpoint, verbose=1, save_best_only=True, save_weights_only=True)

# Training Model

classifier_title.fit(x_data_title, y_data, batch_size=64,
                    epochs=20, callbacks=callback_checkpoint)
classifier_text.fit(x_data_text, y_data, batch_size=64,
                    epochs=20, callbacks=callback_checkpoint)

# Saving Model

classifier_title.save("fake-news-title.h5")
classifier_text.save("fake-news-title.h5")
'''
# Loading Model

classifier_title = load_model("fake-news-title.h5")
# classifier_text = load_model("fake-news-text.h5")

# Prediction

idx = 100

prediction = classifier_title.predict(
    tokenizer_title.string_to_tokens(data_title[idx]))

pred_class_name = CLASS_NAMES[np.argmax(prediction[0])]
orig_class_name = CLASS_NAMES[np.argmax(y_data[idx])]

print("Predicted Category : ", pred_class_name,
      "\nActual Category : ", orig_class_name)

post = 'Netflix is offering 30 days FREE subscription to all its customers, hurry up and join.'
prediction = classifier_title.predict(
    tokenizer_title.string_to_tokens(post))

pred_class_name = CLASS_NAMES[np.argmax(prediction[0])]
print("Predicted : ", pred_class_name)
