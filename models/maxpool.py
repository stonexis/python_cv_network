import numpy as np
from .base_layer import Layer

class MaxPooling(Layer):
    def __init__(self):
        self.name = 'MaxPooling'

    def forward(self, input_data, window_size=2, stride=2):
        dim_data = input_data.shape
        h_out = (dim_data[2] - window_size)//stride + 1
        w_out = (dim_data[3] - window_size)//stride + 1
        out = np.zeros((dim_data[0], dim_data[1], h_out, w_out))
        for b in range(dim_data[0]):
            for c in range(dim_data[1]):
                for i in range(h_out):
                    for j in range(w_out):
                        out[b, c, i, j] = np.max(input_data[b, c, i * stride : i * stride + window_size, j * stride : j * stride + window_size])
        return out
    def grad_x(self):
        pass