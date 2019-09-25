from neuralnetwork.layer import Layer
from neuralnetwork.node import Node, NodeLink
from neuralnetwork.exceptions import IncorrectInputSizeException

class Model:

    def __init__(self, learning_rate = 0.001, momentum = 0.0001):
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def set_input_layer(self, input_layer: Layer):
        self.input_layer = input_layer
        for node in self.input_layer.get_all_nodes():
            inputLink = NodeLink(None, node)
            node.add_previous_link(inputLink)

    def add_hidden_layer(self, hidden_layer: Layer, autoconnect: bool = True):
        self.hidden_layers.append(hidden_layer)
        if (autoconnect):
            hidden_layers_count = len(self.hidden_layers)
            if (hidden_layers_count <= 1):
                self.connect_dense(self.input_layer, self.hidden_layers[0])
            else:
                self.connect_dense(self.hidden_layers[hidden_layers_count - 2], self.hidden_layers[hidden_layers_count - 1])

    def set_output_layer(self, output_layer: Layer, autoconnect: bool = True):
        self.output_layer = output_layer
        if (autoconnect):
            self.connect_dense(self.output_layer, self.hidden_layers[len(self.hidden_layers) - 1])
        for node in output_layer.get_all_nodes():
            outputLink = NodeLink(node)
            node.add_next_link(outputLink)

    def connect_dense(self, prev_layer: Layer, next_layer: Layer):
        prev_nodes = prev_layer.get_all_nodes()
        next_nodes = next_layer.get_all_nodes()
        for prev_node in prev_nodes:
            for next_node in next_nodes:
                this_link = NodeLink(prev_node, next_node)
                prev_node.add_next_link(this_link)
                next_node.add_previous_link(this_link)

    def train(self, dataset = [], batch_size = 5, epoch = 5):
        pass
    
    def test(self, input_values = []):
        pass

    def predict(self, input_values = []):
        pass

    def feed_forward(self, input_values = []):
        if (len(input_values) != self.input_layer.get_node_count()):
            raise IncorrectInputSizeException("Input values supplied doesn't match input layer topology!")
        current_output = self.input_layer.feed_forward(input_values)
        for layer in self.hidden_layers:
            current_output = layer.feed_forward(current_output)
        output_value = self.output_layer.feed_forward(current_output)
        # Do feed_forward on all input nodes
        return output_value

    def backpropagation(self, output_value: float):
        # Calculate deltas on all nodes
        pass

    def update_weight(self):
        # Update weight on all nodes
        pass