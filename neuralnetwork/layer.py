from neuralnetwork.node import Node
from neuralnetwork.exceptions import IncorrectInputSizeException

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
        print("L")
        if input_values:
            if (len(input_values) != self.get_node_count()):
                raise IncorrectInputSizeException("Input values supplied doesn't match input layer topology!")
        iter = 0
        if input_values:
            for node in self.nodes:
                node.feed_forward(input_values[iter])
                iter += 1
        else:
            for node in self.nodes:
                node.feed_forward()
                iter += 1
        
