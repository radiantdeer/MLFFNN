from utilities.operators import sigmoid

# A representation of a Node (that should be a part of a Layer)
class Node:

    def __init__(self):
        self.prev_link = []
        self.next_link = []
        self.output_queue = []
        self.bias = 1
        self.bias_weight = 0
        self.previous_delta_bias = 0
        self.current_delta = 0

    def get_output(self, i = 0):
        return self.output_queue[i]

    def pop_output(self):
        return self.output_queue.pop()

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
                prev_node_value = link.get_prev_node().get_output(0)
                prev_node_weight = link.get_weight()
                input += (prev_node_weight * prev_node_value)
        current_value = sigmoid(input)
        self.output_queue.append(current_value)
        return current_value

    # delta == expected output, say, for an output node. 
    # If delta not supplied, this node will try to grab from it's next node
    def backpropagation(self, delta = None):
        current_value = self.output_queue[0]
        if (delta == None):
            this_node_delta =  current_value * (1 -  current_value)
            next_link_deltas = 0
            for link in self.next_link:
                next_link_deltas += (link.get_weight() * link.get_next_node().get_current_delta())
            self.current_delta = this_node_delta * next_link_deltas
        else:
            self.current_delta = current_value * (1 -  current_value) * (delta -  current_value)
        #print(self.current_delta)

    # Only updates previous links only
    def update_weight(self, learning_rate, momentum, override_x_value = None):
        delta_bias_weight = learning_rate * self.current_delta * self.bias
        delta_bias_weight += momentum * self.previous_delta_bias
        self.bias_weight += delta_bias_weight
        self.previous_delta_bias = delta_bias_weight 
        for link in self.prev_link:
            link.update_weight(learning_rate, momentum, override_x_value)


class NodeLink:

    def __init__(self, prev_node = None, next_node = None, weight = 0):
        self.prev_node = prev_node
        self.next_node = next_node
        self.weight = 0
        self.previous_delta_weight = 0

    def get_prev_node(self):
        return self.prev_node

    def get_next_node(self):
        return self.next_node

    def get_weight(self):
        return self.weight
    
    def set_weight(self, weight: float):
        self.weight = weight
        self.previous_delta_weight = weight

    def update_weight(self, learning_rate, momentum, override_x_value = None):
        if (override_x_value is not None):
            delta_weight = learning_rate * self.next_node.get_current_delta() * override_x_value
        else:
            delta_weight = learning_rate * self.next_node.get_current_delta() * self.prev_node.get_output(0)
        delta_weight += momentum * self.previous_delta_weight
        self.weight += delta_weight
        self.previous_delta_weight = delta_weight