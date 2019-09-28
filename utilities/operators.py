from numpy import exp

def sigmoid(x):
    return 1. / (1. + exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
