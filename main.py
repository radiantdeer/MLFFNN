from neuralnetwork.model import Model
from neuralnetwork.layer import Layer
from utilities.preprocess import load_dataset, normalise
import numpy as np


dataset, expected = load_dataset("dataset/weather.arff")
normed_dataset = normalise(dataset)

model = Model(learning_rate=0.1, momentum=0.9)
model.set_input_layer(Layer(4))
model.add_hidden_layer(Layer(4))
model.set_output_layer(Layer(1), True)
model.prepare()

model.train(normed_dataset, expected, 1, 975)
predicted = []
for val in normed_dataset:
    predicted.append(model.predict(val))

print(predicted)

correct_prediction = 0
for i in range(len(predicted)):
    if (expected[i] == predicted[i]):
        correct_prediction += 1

print("Accuracy: ", correct_prediction/len(predicted))