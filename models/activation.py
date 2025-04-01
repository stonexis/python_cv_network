import numpy as np
from models.base_layer import Layer

class ReLU(Layer):
    def __init__(self):
        self.name = 'ReLU'
        self.preactivation = None
    def forward(self, input_data):
        self.preactivation = input_data
        return np.maximum(0, input_data)

    def grad_x(self, input_data=None):
        if input_data is None:
            input_data = self.preactivation
        J = np.empty((input_data.shape[0], input_data.shape[1], input_data.shape[1]))
        flat_J = np.where(self.preactivation > 0, 1, 0)
        for b in range(flat_J.shape[0]):
            J[b] = np.diag(flat_J[b])
        return J
        