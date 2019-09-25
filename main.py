from neuralnetwork.model import Model
from neuralnetwork.layer import Layer

print("Hello, World!")

model = Model()
model.set_input_layer(Layer(1))
model.add_hidden_layer(Layer(1), True)
model.add_hidden_layer(Layer(1), True)
model.set_output_layer(Layer(1), True)
model.feed_forward([1])