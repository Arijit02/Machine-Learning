import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from os import path
import cv2

filePath = path.dirname(__file__)
datasetDirPath = path.join(filePath, "../../datasets/")
imageDirPath = path.join(filePath, "../../images/")
haarcascadePath = path.join(datasetDirPath,
                            "haarcascades/haarcascade_frontalface_default.xml")
imagePath = path.join(imageDirPath, "man4.jpg")


'''
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dense(units=32, activation="relu"),
    Conv2D(32, (3, 3), activation="relu"),
    Dropout(0.1),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_set = train_datagen.flow_from_directory(path.join(datasetDirPath, "MenWomen/train"),
                                              batch_size=16,
                                              target_size=(64, 64),
                                              class_mode="binary")


test_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_set = test_datagen.flow_from_directory(path.join(datasetDirPath, "MenWomen/test"),
                                            batch_size=16,
                                            target_size=(64, 64),
                                            class_mode="binary")

model.fit_generator(train_set,
                    steps_per_epoch=200,
                    epochs=20,
                    validation_data=test_set,
                    validation_steps=100)


model.save("men-women-2.h5")'''

model = keras.models.load_model("men-women.h5")

'''
test_image = image.load_img(imagePath, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
# print(train_set.class_indices)

if result[0][0] >= 0.5 and result[0][0] <= 1:
    prediction = "Female"
if result[0][0] >= 0 and result[0][0] < 0.5:
    prediction = "Male"

print(prediction)'''


def detector(img, pred):
    cascade = cv2.CascadeClassifier(haarcascadePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 4, cv2.LINE_4)
    cv2.putText(img, pred, (x+int(w/2)-20, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    return img


def genderPredict(Path):
    img = image.load_img(imagePath, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)

    if result[0][0] >= 0.5 and result[0][0] <= 1:
        prediction = "Female"
    if result[0][0] >= 0 and result[0][0] < 0.5:
        prediction = "Male"

    return prediction


img = cv2.imread(imagePath, cv2.IMREAD_COLOR)

prediction = genderPredict(imagePath)

img = detector(img, prediction)

cv2.imshow("Gender Detector", img)

k = cv2.waitKey(0)

if k == ord('q'):
    cv2.destroyAllWindows()
    exit()
