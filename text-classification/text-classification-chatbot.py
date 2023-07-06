import tensorflow as tf
from tensorflow import keras
import numpy as np

''' Encodes received text for the model '''
def encodeText(text, wordIndex):
    encodedText = [1]

    for word in text:
        if word.lower() in wordIndex:
            encodedText.append(wordIndex[word.lower()])
        else:
            encodedText.append(2)

    return encodedText

''' Returns a human readable text for the given review result (POSITIVE/NEGATIVE) '''
def parsePredictionValue(value):
    if value > 0.5: return ("POSITIVE", value * 100)
    return ("NEGATIVE", (1 - value) * 100)


''' Load model & data '''
model = keras.models.load_model("model.h5")
wordIndex = keras.datasets.imdb.get_word_index()

''' Preprocessing '''
wordIndex = {k:(v+3) for k, v in wordIndex.items()}
wordIndex["<PAD>"] = 0
wordIndex["<START>"] = 1
wordIndex["<UNK>"] = 2
wordIndex["<UNUSED>"] = 3

print()
print("** Positive/Negative comments clasification IA **")
print("=================================================")
print("- type 'EXIT' to finish")

while(1):

    ''' Get a phrase to evaluate '''
    print()
    print("Make a comment: ", end="")

    inputText = input()
    '''print()'''

    if inputText.upper() == "EXIT": break

    inputText = inputText.replace(",", "").replace(".", "")
    inputText = inputText.replace("()", "").replace(")", "")
    inputText = inputText.replace(":", "").replace("\"", "")
    inputText = inputText.strip()
    inputText = inputText.split(" ")

    encodedText = encodeText(inputText, wordIndex)
    encodedText = keras.preprocessing.sequence.pad_sequences([encodedText], value=wordIndex["<PAD>"], padding="post", maxlen=250)
    '''print(encodedText)'''

    ''' Making predictions '''
    predictions = model.predict(encodedText, verbose=0)
    prediction = predictions[0][0]

    ''' Show results '''
    (predictedText, confidence) = parsePredictionValue(prediction)

    print(f"Prediction value: \t { format(prediction, '.4f') }")
    print(f"Evaluation result: \t This is a { predictedText } comment (confidence { format(confidence, '.1f') }%).")
