# A representation of a single layer
class Layer:

    def __init__(self, nodes = []):
        self.nodes = nodes
        
    def addNode(self, node):
        self.nodes.append(node)
