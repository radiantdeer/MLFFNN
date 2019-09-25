from neuralnetwork.model import Model
from neuralnetwork.layer import Layer

print("Hello, World!")

model = Model(1, 0.01)
model.set_input_layer(Layer(4))
model.add_hidden_layer(Layer(1), True)
model.add_hidden_layer(Layer(1), True)
model.set_output_layer(Layer(1), True)
model.prepare()

model.train([[1, 2, 3, 4]], [[1]], 1, 10)