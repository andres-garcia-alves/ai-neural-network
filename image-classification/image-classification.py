import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

''' Load and split data (60000:10000) '''
(trainImages, trainLabels), (testImages, testLabels) = keras.datasets.fashion_mnist.load_data()

''' Category names '''
classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Abkle Boot"]

''' Normalize image data between 0-1 '''
trainImages = trainImages/255.0
testImages = testImages/255.0

''' Transformation pipeline '''
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

''' Build model '''
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(trainImages, trainLabels, epochs=10)

''' Check model accuracy '''
testLoss, testAccuracy = model.evaluate(testImages, testLabels)
print()
print("\n\rModel accuracy:", testAccuracy)

''' Making predictions '''
predictions = model.predict(testImages)
'''predictions = model.predict([testImages[0]])'''
print()

for i in range(5):
    predictionValue = np.argmax(predictions[i])
    print(f"Image #0{i} - Predicted: { classNames[predictionValue] }, Actual: { classNames[testLabels[i]] }")

    plt.grid(False)
    plt.title(f"Prediction: { classNames[predictionValue] } - Actual: { classNames[testLabels[i]] }")
    plt.imshow(testImages[i], cmap=plt.cm.binary)
    plt.show()
