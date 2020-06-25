from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import pandas as pd
import numpy as np
from os import path

filePath = path.dirname(__file__)

'''
classifier = keras.models.Sequential()

classifier.add(keras.layers.Conv2D(
    32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(keras.layers.Conv2D(
    32, (3, 3), activation='relu'))
classifier.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(keras.layers.Flatten())
classifier.add(keras.layers.Dense(units=128, activation='relu'))
classifier.add(keras.layers.Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory(path.join(filePath, '../../datasets/cats_and_dogs_filtered/train/'),
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
testing_set = test_datagen.flow_from_directory(path.join(filePath, '../../datasets/cats_and_dogs_filtered/validation/'),
                                               target_size=(64, 64),
                                               batch_size=16,
                                               class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=175,
                         epochs=20,
                         validation_data=testing_set,
                         validation_steps=1000)

classifier.save("Cat-Dog.h5")'''

classifier = keras.models.load_model("Cat-Dog.h5")

test_image = image.load_img(
    path.join(filePath, '../../images/dog1.jpg'), target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
# print(result)
if result[0][0] <= 1 and result[0][0] > 0.5:
    prediction = 'dog'
if result[0][0] >= 0 and result[0][0] <= 0.5:
    prediction = 'cat'

print(prediction)
