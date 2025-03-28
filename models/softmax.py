import numpy as np
from models.base_layer import Layer

class Softmax(Layer):
    def __init__(self):
        self.name = 'Softmax'
    def forward(self, input_data):
        exp_data = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.probs = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        return self.probs
    def grad_x(self, input_data):
        pass