# import tensorflow as tf
import numpy as np
import pandas as pd
# import sklearn
from sklearn import linear_model, model_selection
# from sklearn.utils import shuffle
from os import path
from matplotlib import pyplot as plt
import pickle
from matplotlib import style

file_path = path.dirname(__file__)
csv_path = path.join(file_path, "./datasets/student/student-mat.csv")
data = pd.read_csv(csv_path, sep=";")         # sep -> separator
# print(data.head())                          # prints first 5 data elements
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, Y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.1)

    model = linear_model.LinearRegression()

    model.fit(x_train, y_train)                     # Training
    acc = model.score(x_test, y_test)               # Testing
    print(acc)

    if acc > best:
        best = acc
        with open("student.pickel", "wb") as f:
            pickle.dump(model, f)'''

pickle_in = open("student.pickel", "rb")
model = pickle.load(pickle_in)

# Coefficient of best fit line
# print("Coefficient: " + str(model.coef_))

# Intercept of best fit line
# print("Intercept: " + str(model.intercept_))

predictions = model.predict(x_test)

for prediction in range(len(predictions)):
    # print(predictions[prediction], x_test[prediction], y_test[prediction])
    pass

    # score = sklearn.metrics.accuracy_score(y_test, predictions)


p = "absences"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
