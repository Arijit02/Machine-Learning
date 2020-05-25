from sklearn import datasets, svm, model_selection, metrics, neighbors
import numpy as np

cancer = datasets.load_breast_cancer()
# print(cancer.keys())
# print(type(cancer))
# print(cancer['feature_names'])
# print(cancer['target_names'])

X = cancer['data']
Y = cancer['target']

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, Y, test_size=0.15
)

model = svm.SVC(kernel="linear", C=1)
# model = neighbors.KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
# acc = model.score(x_test, y_test)
# print(acc)

y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
