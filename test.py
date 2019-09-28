from numpy import *
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(4, 8), alpha=0.1, learning_rate_init=0.2, momentum=0.9, activation="logistic", solver="sgd", max_iter=1000, batch_size=3)

dataset = [
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
    ]
expected = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0]

print(model.fit(dataset, expected))
print(model.predict(dataset))
