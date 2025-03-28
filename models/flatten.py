import numpy as np
from models.base_layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        self.name = 'Flatten'

    def forward(self, input_data):
        return np.reshape(input_data, (input_data.shape[0], -1))

    def grad_x(self):
        pass