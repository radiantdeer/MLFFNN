from neuralnetwork.model import Model
from neuralnetwork.layer import Layer
from utilities.preprocess import load_dataset, normalise
import numpy as np


dataset, expected = load_dataset("dataset/weather.arff")
normed_dataset = normalise(dataset)

model = Model(hidden_layer=1, nb_nodes=[4], learning_rate=0.1, momentum=0.9)

model.train(normed_dataset, expected, 1, 975)
predicted = []
for val in normed_dataset:
    predicted.append(model.predict(val))

print(predicted)

correct_prediction = 0
for i in range(len(predicted)):
    if (expected[i] == predicted[i]):
        correct_prediction += 1

print("Accuracy: ", round(correct_prediction/len(predicted)*100, 4))