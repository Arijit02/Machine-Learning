from sklearn import model_selection, preprocessing, tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# import numpy as np
from os import path

file_path = path.dirname(__file__)
data_path = path.join(file_path, "../datasets/car.data")

df = pd.read_csv(data_path)
# print(df.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(df['buying']))
maint = le.fit_transform(list(df['maint']))
doors = le.fit_transform(list(df['doors']))
persons = le.fit_transform(list(df['persons']))
lug_boot = le.fit_transform(list(df['lug_boot']))
safety = le.fit_transform(list(df['safety']))
class_name = le.fit_transform(list(df['class']))

predict = 'class'

X = list(zip(buying, maint, doors, persons, lug_boot, safety))
Y = list(class_name)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, Y, test_size=0.1
)

# model = KNeighborsClassifier()
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
# print(acc)

predictions = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(x_test)):
    print("Predicted: ", names[predictions[x]], "Data: ",
          x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
