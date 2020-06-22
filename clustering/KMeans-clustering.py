from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import load_digits
from sklearn import metrics
from os import path
import cv2

file_path = path.dirname(__file__)
image_path = path.join(file_path, "../images/six.png")
# image_path = path.join(file_path, "../images/four.png")

img = cv2.imread(image_path, 0)
img = scale(img)
img.resize((64,))
# print(img.shape)
# print(img)
# cv2.imshow("image", img)
# cv2.waitKey(0)


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print("%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.accuracy_score(y, estimator.labels_)
             )
          )


digits = load_digits()
x = scale(digits['data'])
y = digits['target']

# print(x[0].shape)
# print(y)

k = len(np.unique(y))
samples, features = digits['data'].shape

clf = KMeans(n_clusters=k, init="random", n_init=10, max_iter=600)
bench_k_means(clf, "1", x)
y_pred = clf.predict([img])
print(y_pred)
