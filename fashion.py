# import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# print(test_images[0])

# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

# print(test_acc, test_loss)

predictions = model.predict(test_images)

i = 100

for j in range(i, i+10):
    index = np.argmax(predictions[j])
    style.use('ggplot')
    plt.grid(False)
    plt.subplot(2, 5, j-i+1), plt.imshow(test_images[j], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[j]])
    plt.title(class_names[index])


plt.show()
