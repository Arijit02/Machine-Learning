from sklearn import preprocessing, model_selection, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from os import path

file_path = path.dirname(__file__)
csv_path = path.join(file_path, "../datasets/weight-height.csv")

df = pd.read_csv(csv_path)
# print(df.head())

le = preprocessing.LabelEncoder()

Gender = le.fit_transform(list(df['Gender']))

predict = 'Gender'
X = np.array(df.drop([predict], 1))
Y = list(Gender)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, Y, test_size=0.2
)

# model = linear_model.LinearRegression()
# model = DecisionTreeClassifier()
model = KNeighborsClassifier(n_neighbors=11)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
# print(acc)

'''
predictions = model.predict(x_test)
headers = ["Female", "Male"]

for x in range(len(x_test)):
    print("Predicted: ", headers[predictions[x]],
          "Features: ", x_test[x], "Actual: ", headers[y_test[x]])'''

height = float(input("Enter your height (in feet): "))
weight = float(input("Enter your weight (in kgs): "))

height *= 12
weight /= 0.453592

prediction = model.predict(np.array([[height, weight]]))
headers = ["Female", "Male"]

print("Predicted: ", headers[prediction[0]])
