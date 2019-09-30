from neuralnetwork.node import Node
from neuralnetwork.exceptions import IncorrectSizeException

# A representation of a single layer
class Layer:

    def __init__(self, num_nodes = 1):
        self.nodes = []
        for i in range(num_nodes):
            self.nodes.append(Node())

    def add_node(self, node):
        self.nodes.append(node)

    def get_node_count(self):
        return len(self.nodes)

    # Returning None if position is out of bounds
    def get_node_at(self, position):
        node = None
        try:
            node = self.nodes[position]
        except:
            pass
        return node

    def get_all_nodes(self):
        return self.nodes

    def feed_forward(self, input_values = None):
        iter = 0
        output_values = [0 for x in range(self.get_node_count())]

        if (input_values is not None):
            if (len(input_values) != self.get_node_count()):
                raise IncorrectSizeException("Input values supplied doesn't match input layer topology!")
            for node in self.nodes:
                output_values[iter] = node.feed_forward(input_values[iter])
                iter += 1
        else:
            for node in self.nodes:
                output_values[iter] = node.feed_forward()
                iter += 1
        return output_values

    def backpropagation(self, real_values = None):
        if real_values is not None:
            if(real_values.shape[0] != self.get_node_count()):
                raise IncorrectSizeException("Values supplied doesn't match output layer topology!")
            iter = 0
            for node in self.nodes:
                node.backpropagation(real_values[iter])
                iter += 1
        else:
            for node in self.nodes:
                node.backpropagation()
        #print()

    # Only updates previous links only
    def temporary_update_weight(self, learning_rate, momentum, override_x_values = None):
        if override_x_values is not None:
            iter = 0
            for node in self.nodes:
                node.temporary_update_weight(learning_rate, momentum, override_x_values[iter])
                iter += 1
        else:
            for node in self.nodes:
                node.temporary_update_weight(learning_rate, momentum)

    def update_weight(self):
        for node in self.nodes:
            node.update_weight()

