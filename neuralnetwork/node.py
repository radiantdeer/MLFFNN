from utilities.operators import sigmoid

# A representation of a Node (that should be a part of a Layer)
class Node:

    def __init__(self):
        self.prev_link = []
        self.next_link = []
        self.current_value = 0
        self.bias = 0

    def add_previous_link(self, prev_link):
        self.prev_link.append(prev_link)

    def add_next_link(self, next_link):
        self.next_link.append(next_link)

    def feed_forward(self, input_value = None):
        input = self.bias
        print("N", end="")
        if input_value:
            input += (self.prev_link[0].get_weight() * input_value)
        else:
            for link in self.prev_link:
                prev_node_value = link.prev.current_value
                prev_node_weight = link.get_weight()
                input += (prev_node_weight * prev_node_value)
        self.current_value = sigmoid(input)

    def backpropagation(self, delta):
        # Will implement later. Basically will just pass delta to all of it's connected Links
        pass

    def update_weight(self, delta):
        # Will implement later. Basically will just pass delta to all of it's connected Links
        pass


class NodeLink:

    def __init__(self, prev: Node = None, next: Node = None, weight: float = 0):
        self.prev = prev
        self.next = next
        self.weight = 0
        self.currentDelta = 0

    def set_weight(self, weight: float):
        self.weight = weight

    def get_weight(self):
        return self.weight

    def feed_forward(self, input_value: float):
        # Will implement later
        pass

    def backpropagation(self, delta: float):
        # Will implement later
        pass

    def update_weight(self, delta):
        # Will implement later
        pass