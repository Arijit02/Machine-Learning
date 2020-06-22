from os import path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
import nltk
pd.options.mode.chained_assignment = None
# nltk.download('punkt')

stemmer = LancasterStemmer()


def words_array_maker(column):
    words = []
    for instance in column:
        instance = str(instance)
        instance.replace(":", "").replace(";", "").replace("\'", "").replace(
            ",", "").replace(".", "").replace("?", "").replace("[", "").replace("]", "")
        wrds = nltk.word_tokenize(instance)
        words.extend(wrds)

    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))
    return words


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words_title))]

    s = str(s)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for wrd in s_words:
        for i, w in enumerate(words):
            if wrd == w:
                bag[i] = 1

    return np.array(bag)


BASE_DIR = path.dirname(__file__)
CSV_PATH = path.join(BASE_DIR, "../datasets/fake.csv")
data = pd.read_csv(CSV_PATH, sep=",", header=0)
dataset = data.copy()

FEATURES = ["author", "published", "title", "language",
            "country", "spam_score", "replies_count", "likes", "comments", "shares"]

LABELS = ["type"]

CATEGORICAL = ["author", "published", "language", "country", "type"]

le = preprocessing.LabelEncoder()

for feature in CATEGORICAL:
    dataset[feature] = le.fit_transform(list(dataset[feature]))

words_title = words_array_maker(dataset['title'])
# words_text = words_array_maker(dataset['text'])


for i, title in enumerate(dataset['title']):
    bag = bag_of_words(title, words_title)
    dataset['title'][i] = bag
    # if i == 2:
    # break

'''for i, text in enumerate(dataset['text']):
    bag = bag_of_words(text, words_text)
    print(dataset['text'][i])'''


X = np.array(dataset[FEATURES])
Y = np.ravel(np.array(dataset[LABELS]))

# print(Y.shape)
# print(X.shape)


# print(X[:][2].shape)
# print(Y.head())

'''
CLASS_NAMES = ['bias', 'bs', 'conspiracy',
               'fake', 'hate', 'junksci', 'satire', 'state']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=100)

# model = DecisionTreeClassifier(criterion="entropy")
model = RandomForestClassifier(n_jobs=2, random_state=0)
model.fit(X_train, Y_train)

results = model.score(X_test, Y_test)
print(results)

# predictions = model.predict(X_test)

# for x in range(len(X_test)):
#     print("Predicted: ", CLASS_NAMES[predictions[x]],
#           " Actual: ", CLASS_NAMES[Y_test[x][0]])
'''
