from copy import deepcopy
import numpy as np
#from utilities.operators import sigmoid

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:

    def __init__(self, learning_rate = 0.01, momentum = 0.001):
        self.num_nodes_each_layer = [0, 0]
        self.weights = []
        self.previous_weights = []
        self.layer_deltas = []
        self.input_vals = []
        self.layer_results = []
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.ready = False

    def add_input_layer(self, num_of_nodes):
        self.num_nodes_each_layer[0] = num_of_nodes

    def add_layer(self, num_of_nodes):
        self.num_nodes_each_layer.insert(len(self.num_nodes_each_layer) - 1, num_of_nodes)

    def add_output_layer(self, num_of_nodes):
        self.num_nodes_each_layer[len(self.num_nodes_each_layer) - 1] = num_of_nodes

    def prepare(self):
        for iter in range(len(self.num_nodes_each_layer) - 1):
            this_weights = np.random.rand(self.num_nodes_each_layer[iter], self.num_nodes_each_layer[iter + 1])
            this_weights = this_weights / (self.num_nodes_each_layer[iter] * 2)
            self.weights.append(this_weights)
            self.previous_weights.append(np.zeros((self.num_nodes_each_layer[iter], self.num_nodes_each_layer[iter + 1])))
        print(self.weights)
        self.ready = True

    def train(self, dataset, expected, batch_size = 5, epoch = 5):
        if (batch_size > len(expected)):
            raise Exception("Batch size is greater than data provided!")
        for i in range(epoch):
            padding = 0
            print ("Epoch #" + str(i + 1))
            when_to_backprop = batch_size
            for iter in range(len(dataset)):
                result = self.feed_forward(dataset[iter])
                when_to_backprop -= 1
                if (when_to_backprop == 0):
                    while(when_to_backprop < batch_size):
                        self.backpropagation(expected[padding])
                        self.update_weight()
                        padding += 1
                        when_to_backprop += 1
        
            if (when_to_backprop != 0):
                while(when_to_backprop < batch_size):
                    self.backpropagation(expected[padding])
                    self.update_weight()
                    padding += 1
                    when_to_backprop += 1

    def feed_forward(self, input_values):
        if self.ready:
            self.input_vals.append(input_values)
            self.layer_results.append([])
            p = len(self.layer_results) - 1
            result = sigmoid(np.dot(input_values, self.weights[0]))
            self.layer_results[p].append(result)
            for iter in range(1, len(self.weights)):
                result = sigmoid(np.dot(result, self.weights[iter]))
                self.layer_results[p].append(result)
            self.layer_results[p] = np.array(self.layer_results[p])
            return result

    def backpropagation(self, actual_values):    
        self.layer_deltas = [[] for x in range(len(self.layer_results[0]))]
        p = len(self.layer_results[0]) - 1

        # Output layer
        this_layer_delta = []   
        for iter in range(len(self.layer_results[0][len(self.layer_results[0]) - 1])):
            output = self.layer_results[0][len(self.layer_results[0]) - 1][iter]
            delta = sigmoid_derivative(output) * (actual_values[iter] - output)
            this_layer_delta.append(delta)
        this_layer_delta = np.array(this_layer_delta).T
        self.layer_deltas[p] = this_layer_delta

        # Remaining hidden layers
        for backwards_layer_iter in range(len(self.layer_results[0]) - 2, -1, -1):
            this_layer_output = self.layer_results[0][backwards_layer_iter]
            this_layer_next_weights = self.weights[backwards_layer_iter + 1]
            next_layer_delta = self.layer_deltas[backwards_layer_iter + 1]
            this_layer_delta = []
            for iter in range(len(this_layer_output)):
                node_delta = sigmoid_derivative(this_layer_output[iter])
                cumulative_next_delta = 0
                for iter2 in range(len(this_layer_next_weights[iter])):
                    cumulative_next_delta += this_layer_next_weights[iter][iter2] * next_layer_delta[iter2]
                node_delta *= cumulative_next_delta
                this_layer_delta.append(node_delta)
            this_layer_delta = np.array(this_layer_delta).T
            self.layer_deltas[backwards_layer_iter] = this_layer_delta

    def update_weight(self):
        # First set of weights (from input/first hidden layer to the next one)
        temp = deepcopy(self.previous_weights)
        self.previous_weights = deepcopy(self.weights)
        i, j = self.weights[0].shape
        for x in range(i):
            for y in range(j):
                self.weights[0][x][y] += (self.learning_rate * self.layer_deltas[0][y] * self.input_vals[0][y]) + (self.momentum * self.previous_weights[0][x][y])

        # Remaining sets of weights
        for iter in range(1, len(self.weights)):
            i, j = self.weights[iter].shape
            for x in range(i):
                for y in range(j):
                    self.weights[iter][x][y] += (self.learning_rate * self.layer_deltas[iter][y] * self.layer_results[0][iter][y]) + (self.momentum * self.previous_weights[iter][x][y])
        self.layer_results.pop(0)
        self.input_vals.pop(0)

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

model = NeuralNetwork(learning_rate=0.00001, momentum=0.00005)
model.add_input_layer(4)
model.add_layer(4)
model.add_layer(4)
model.add_output_layer(1)
model.prepare()

model.train(normed_dataset, expected, batch_size=13, epoch=10)
for val in normed_dataset:
    print(model.feed_forward(val))