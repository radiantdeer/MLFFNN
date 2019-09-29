from neuralnetwork.layer import Layer
from neuralnetwork.node import Node, NodeLink
from neuralnetwork.exceptions import IncorrectSizeException, ModelIsNotReadyException, LayerMissingException
import numpy as np
from utilities.operators import sigmoid, sigmoid_derivative

class Model:

    def __init__(self, learning_rate = 0.001, momentum = 0.0001):
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.ready_for_training = False
    
    def set_input_layer(self, input_layer: Layer):
        self.input_layer = input_layer
        for node in self.input_layer.get_all_nodes():
            inputLink = NodeLink(None, node)
            node.add_previous_link(inputLink)

    def add_hidden_layer(self, hidden_layer: Layer, autoconnect: bool = True):
        if (self.input_layer == None):
            raise LayerMissingException("Please define an input layer first!")

        self.hidden_layers.append(hidden_layer)
        if (autoconnect):
            hidden_layers_count = len(self.hidden_layers)
            if (hidden_layers_count <= 1):
                self.connect_dense(self.input_layer, self.hidden_layers[0])
            else:
                self.connect_dense(self.hidden_layers[hidden_layers_count - 2], self.hidden_layers[hidden_layers_count - 1])

    def set_output_layer(self, output_layer: Layer, autoconnect: bool = True):
        if (self.input_layer == None):
            raise LayerMissingException("Please define an input layer first!")

        self.output_layer = output_layer
        if (autoconnect):
            if (len(self.hidden_layers) > 0):
                self.connect_dense(self.hidden_layers[len(self.hidden_layers) - 1], self.output_layer)
            else:
                self.connect_dense(self.input_layer, self.output_layer)

    def connect_dense(self, prev_layer: Layer, next_layer: Layer):
        prev_nodes = prev_layer.get_all_nodes()
        next_nodes = next_layer.get_all_nodes()
        for prev_node in prev_nodes:
            for next_node in next_nodes:
                this_link = NodeLink(prev_node, next_node)
                prev_node.add_next_link(this_link)
                next_node.add_previous_link(this_link)

    def prepare(self):
        if (self.input_layer == None):
            raise LayerMissingException("No input layer defined!")
        elif (self.output_layer == None):
            raise LayerMissingException("No output layer defined!")
        
        for node in self.input_layer.get_all_nodes():
            num_of_weights = 2 + len(node.get_all_prev_links())
            node.set_bias_weight(np.random.rand() / num_of_weights)
            for prev_link in node.get_all_prev_links():
                prev_link.set_weight(np.random.rand() / num_of_weights)

        for layer in self.hidden_layers:
            for node in layer.get_all_nodes():
                num_of_weights = 2 + len(node.get_all_prev_links())
                node.set_bias_weight(np.random.rand() / num_of_weights)
                for prev_link in node.get_all_prev_links():
                    prev_link.set_weight(np.random.rand() / num_of_weights)
        
        for node in self.output_layer.get_all_nodes():
            num_of_weights = 2 + len(node.get_all_prev_links())
            node.set_bias_weight(np.random.rand() / num_of_weights)
            for prev_link in node.get_all_prev_links():
                prev_link.set_weight(np.random.rand() / num_of_weights)

        self.ready_for_training = True

    def train(self, dataset, real_values, batch_size = 5, epoch = 5, verbose = False):
        if not self.ready_for_training:
            raise ModelIsNotReadyException("Model has not been configured properly. (usually solved by calling 'prepare()' first.)")
        
        for x in range(epoch):
            if (verbose):
                print("Epoch #" + str(x + 1))
            batch_result_queue = []
            iter = batch_size
            padding = 0
            for data in dataset:
                output = self.feed_forward(data)
                if (verbose):
                    print(output)
                batch_result_queue.append(output)
                iter -= 1
                if (iter <= 0):
                    for result in batch_result_queue:
                        self.backpropagation(real_values[padding])
                        self.update_weight(dataset[padding])
                        padding += 1
                    batch_result_queue = []
                    iter = batch_size
            
            if (len(batch_result_queue) > 0):
                for result in batch_result_queue:
                    self.backpropagation(real_values[padding])
                    self.update_weight(dataset[padding])
                    padding += 1

    def test(self, dataset, real_value):
        pass

    def predict(self, input_values):
        if (len(input_values) != self.input_layer.get_node_count()):
            raise IncorrectSizeException("Input values supplied doesn't match input layer topology!")
        result = self.feed_forward(input_values)
        if (len(result) < 2):
            if (result[0] > 0.5):
                return 1
            else:
                return 0

    def feed_forward(self, input_values):
        if not self.ready_for_training:
            raise ModelIsNotReadyException("Model has not been configured properly. (usually solved by calling 'prepare()' first.)")
        if (len(input_values) != self.input_layer.get_node_count()):
            raise IncorrectSizeException("Input values supplied doesn't match input layer topology!")
        
        self.input_layer.feed_forward(input_values)
        for layer in self.hidden_layers:
            layer.feed_forward()
        output_value = self.output_layer.feed_forward()

        return output_value

    def backpropagation(self, output_values):
        self.output_layer.backpropagation(output_values)
        num_hidden_layers = len(self.hidden_layers)
        for i in range(num_hidden_layers - 1, -1, -1):
            self.hidden_layers[i].backpropagation()
        self.input_layer.backpropagation()

    def update_weight(self, real_values):
        self.input_layer.update_weight(self.learning_rate, self.momentum, real_values)
        for layer in self.hidden_layers:
            layer.update_weight(self.learning_rate, self.momentum)
        self.output_layer.update_weight(self.learning_rate, self.momentum)