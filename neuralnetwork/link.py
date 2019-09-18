# A representation of the link between nodes
class Link:

    def __init__(self, prev = None, next = None, weight = 0):
        self.prev = prev
        self.next = next
        self.weight = 0

    def set_weight(self, weight):
        self.weight = weight

    def backpropagation(self, delta):
        # Will implement later
        pass

    def update_weight(self, delta):
        # Will implement later
        pass