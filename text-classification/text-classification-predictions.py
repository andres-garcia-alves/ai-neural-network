import tensorflow as tf
from tensorflow import keras
import numpy as np

''' Returns a human readable text for the given review '''
def decodeReview(text, reverseWordIndex):
    return " ".join([reverseWordIndex.get(i, "?") for i in text])

''' Returns a human readable text for the given review result (POSITIVE/NEGATIVE) '''
def parsePredictionValue(value):
    if value > 0.5: return "POSITIVE"
    return "NEGATIVE"


''' Load and split data '''
(trainText, trainLabels), (testText, testLabels) = keras.datasets.imdb.load_data(num_words=10000)

''' Preprocessing '''
wordIndex = keras.datasets.imdb.get_word_index()
wordIndex = {k:(v+3) for k, v in wordIndex.items()}
wordIndex["<PAD>"] = 0
wordIndex["<START>"] = 1
wordIndex["<UNK>"] = 2
wordIndex["<UNUSED>"] = 3

trainText = keras.preprocessing.sequence.pad_sequences(trainText, value=wordIndex["<PAD>"], padding="post", maxlen=250)
testText = keras.preprocessing.sequence.pad_sequences(testText, value=wordIndex["<PAD>"], padding="post", maxlen=250, )

reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()])

''' Load model '''
model = keras.models.load_model("model.h5")

''' Making predictions '''
predictions = model.predict(np.array([testText[0]]))
'''predictions = model.predict(testText)'''

prediction = predictions[0][0]
predictedText = parsePredictionValue(prediction)
actualText = parsePredictionValue(testLabels[0])

print()
print("Review #01:")
print(decodeReview(testText[0], reverseWordIndex))
print()
print(f"Review #01 - Predicted: { predictedText } ({ prediction }), Actual: { actualText }.")
