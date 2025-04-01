import numpy as np
from models.base_layer import Layer

class Softmax(Layer):
    def __init__(self):
        self.name = 'Softmax'
        self.probs = None
        self.preactivation = None
    def forward(self, input_data):
        self.preactivation = input_data
        exp_data = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.probs = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        return self.probs
    def grad_x(self, input_data=None):
        if input_data is None:
            input_data = self.preactivation
        jacobian = np.empty((self.probs.shape[0], self.probs.shape[1], self.probs.shape[1]))
        for b in range(self.probs.shape[0]):
            for i in range(self.probs.shape[1]):
                for j in range(self.probs.shape[1]):
                    if i == j:
                        jacobian[b, i, j] = self.probs[b, i] * (1 - self.probs[b,i])
                    else:
                        jacobian[b, i, j] = -self.probs[b, i] * self.probs[b,j]
        return jacobian