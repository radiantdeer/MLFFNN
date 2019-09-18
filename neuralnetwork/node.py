from utilities.operators import sigmoid
from neuralnetwork.link import Link

# A representation of a Node (that should be a part of a Layer)
class Node:

    def __init__(self):
        self.prevLink = []
        self.nextLink = []

    def connectPrev(self, prevNode, weight = 0):
        thisPrevLink = Link(prevNode, self, weight)
        self.prevLink.append(thisPrevLink)

    def connectNext(self, nextNode, weight = 0):
        thisNextLink = Link(self, nextNode, weight)
        self.nextLink.append(thisNextLink)

    def feed_forward(self, input_value):
        return sigmoid(input_value)

    def backpropagation(self, delta):
        # Will implement later. Basically will just pass delta to all of it's connected Links
        pass

    def update_weight(self, delta):
        # Will implement later. Basically will just pass delta to all of it's connected Links
        pass