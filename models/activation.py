import numpy as np
from models.base_layer import Layer

class ReLU(Layer):
    def __init__(self):
        self.name = 'ReLU'
        self.relu_data = None
    def forward(self, input_data):
        self.relu_data = np.maximum(0, input_data)
        return self.relu_data
    def grad_x(self, input_data):
        J = np.empty((input_data.shape[0], input_data.shape[1], input_data.shape[1]))
        flat_J = np.where(self.relu_data > 0, 1, 0)
        for b in range(flat_J.shape[0]):
            J[b] = np.diag(flat_J[b])
        return J
