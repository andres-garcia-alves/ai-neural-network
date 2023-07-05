import tensorflow as tf
from tensorflow import keras
import numpy as np

''' Load and split data (25000:25000) '''
(trainText, trainLabels), (testText, testLabels) = keras.datasets.imdb.load_data(num_words=100000)
wordIndex = keras.datasets.imdb.get_word_index()

''' Preprocessing '''
wordIndex = {k:(v+3) for k, v in wordIndex.items()}
wordIndex["<PAD>"] = 0
wordIndex["<START>"] = 1
wordIndex["<UNK>"] = 2
wordIndex["<UNUSED>"] = 3

trainText = keras.preprocessing.sequence.pad_sequences(trainText, value=wordIndex["<PAD>"], padding="post", maxlen=250)
testText = keras.preprocessing.sequence.pad_sequences(testText, value=wordIndex["<PAD>"], padding="post", maxlen=250)

reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()])

''' Transformation pipeline '''
model = keras.Sequential()
model.add(keras.layers.Embedding(100000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()

''' Build model '''
xValue = trainText[:10000]
yValue = trainLabels[:10000]
xTrain = trainText[10000:]
yTrain = trainLabels[10000:]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(xTrain, yTrain, epochs=40, batch_size=512, validation_data=(xValue, yValue), verbose=1)

''' Check model accuracy '''
testLoss, testAccuracy = model.evaluate(testText, testLabels)
print()
print(f"Model accuracy: { format(testAccuracy, '.4f') }")

''' Save model for futher usage '''
model.save("model.h5")
print(f"Model saved: 'model.h5'")
