from utilities.operators import sigmoid

# A representation of a Node (that should be a part of a Layer)
class Node:

    def __init__(self):
        self.prev_link = []
        self.next_link = []
        self.current_output = 0
        self.bias = 1
        self.bias_weight = 0
        self.temporary_bias_weight = 0
        self.previous_delta_bias = 0
        self.current_delta = 0

    def get_current_output(self):
        return self.current_output

    def get_all_prev_links(self):
        return self.prev_link

    def get_all_next_links(self):
        return self.next_link

    def get_bias(self):
        return self.bias

    def get_bias_weight(self):
        return self.bias_weight

    def get_current_delta(self):
        return self.current_delta

    def set_bias(self, bias: float):
        self.bias = bias

    def set_bias_weight(self, weight: float):
        self.bias_weight = weight

    def add_previous_link(self, prev_link):
        self.prev_link.append(prev_link)

    def add_next_link(self, next_link):
        self.next_link.append(next_link)

    def feed_forward(self, input_value = None):
        input = (self.bias * self.bias_weight)
        if (input_value != None):
            input += (self.prev_link[0].get_weight() * input_value)
        else:
            for link in self.prev_link:
                prev_node_value = link.get_prev_node().get_current_output()
                prev_node_weight = link.get_weight()
                input += (prev_node_weight * prev_node_value)
        self.current_output = sigmoid(input)
        return self.current_output

    # delta == expected output, say, for an output node.
    # If delta not supplied, this node will try to grab from it's next node
    def backpropagation(self, delta = None):
        if (delta == None):
            this_node_delta = self.current_output * (1 - self.current_output)
            next_link_deltas = 0
            for link in self.next_link:
                next_link_deltas += (link.get_temporary_weight() * link.get_next_node().get_current_delta())
            self.current_delta = this_node_delta * next_link_deltas
        else:
            self.current_delta = self.current_output * (1 - self.current_output) * (delta - self.current_output)

    # Only updates previous links only
    def temporary_update_weight(self, learning_rate, momentum, override_x_value = None):
        delta_bias_weight = learning_rate * self.current_delta * self.bias
        delta_bias_weight += momentum * self.previous_delta_bias
        self.temporary_bias_weight += delta_bias_weight
        self.previous_delta_bias = delta_bias_weight
        for link in self.prev_link:
            link.temporary_update_weight(learning_rate, momentum, override_x_value)

    # Only updates previous links only. This is the one that will be called after the end of a mini-batch
    def update_weight(self):
        self.bias_weight = self.temporary_bias_weight
        for link in self.prev_link:
            link.update_weight()

class NodeLink:

    def __init__(self, prev_node = None, next_node = None, weight = 0):
        self.prev_node = prev_node
        self.next_node = next_node
        self.weight = 0
        self.temporary_weight = 0
        self.previous_delta_weight = 0

    def get_prev_node(self):
        return self.prev_node

    def get_next_node(self):
        return self.next_node

    def get_weight(self):
        return self.weight

    def get_temporary_weight(self):
        return self.temporary_weight

    def set_weight(self, weight: float):
        self.weight = weight
        self.previous_delta_weight = weight
        self.temporary_weight = weight

    def temporary_update_weight(self, learning_rate, momentum, override_x_value = None):
        if (override_x_value is not None):
            delta_weight = learning_rate * self.next_node.get_current_delta() * override_x_value
        else:
            delta_weight = learning_rate * self.next_node.get_current_delta() * self.prev_node.get_current_output()
        delta_weight += momentum * self.previous_delta_weight
        self.temporary_weight += delta_weight
        self.previous_delta_weight = delta_weight

    def update_weight(self):
        self.weight = self.temporary_weight
