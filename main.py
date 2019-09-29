from neuralnetwork.model import Model
from neuralnetwork.layer import Layer
import numpy as np

# sunny = 1    | FALSE = 0 | NO = 0
# overcast = 2 | TRUE = 1 | YES = 1
# rainy = 3    |
dataset = np.array([
    [1, 85, 85, 0],
    [1, 80, 90, 1],
    [2, 83, 96, 0],
    [3, 70, 96, 0],
    [3, 68, 80, 0],
    [3, 65, 70, 1],
    [2, 64, 65, 1],
    [1, 72, 95, 0],
    [1, 69, 70, 0],
    [3, 75, 80, 1],
    [2, 72, 90, 1],
    [2, 81, 75, 0],
    [3, 71, 91, 1]
    ])
expected = np.array([[0], [0], [1], [1], [1], [0], [1], [0], [1], [1], [1], [1], [0]])

# Preprocessing dataset
normed_dataset = dataset / dataset.max(axis=0)

print(normed_dataset)

model = Model(learning_rate=0.1, momentum=0.9)
model.set_input_layer(Layer(4))
model.add_hidden_layer(Layer(4))
model.set_output_layer(Layer(1), True)
model.prepare()

model.train(normed_dataset, expected, 1, 950)
predicted = []
for val in normed_dataset:
    predicted.append(model.predict(val))

print(predicted)

correct_prediction = 0
for i in range(len(predicted)):
    if (expected[i] == predicted[i]):
        correct_prediction += 1

print("Accuracy: ", correct_prediction/len(predicted))