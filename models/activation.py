import numpy as np
from models.base_layer import Layer

class ReLU(Layer):
    def __init__(self):
        self.name = 'ReLU'
    def forward(self, input_data):
        return np.maximum(0, input_data)
    def grad_x(self, input_data):
        pass