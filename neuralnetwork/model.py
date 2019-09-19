from layer import Layer

class Model:

    def __init__(self, input_layer = Layer(), hidden_layers = Layer(), output_layer = Layer()):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
    
    def feed_forward(self, input_values = []):
        output_value = 0
        # Do feed_forward on all input nodes
        return output_value

    def backpropagation(self, output_value, label_value):
        # Calculate deltas on all nodes
        pass

    def update_weight(self):
        # Update weight on all nodes
        pass