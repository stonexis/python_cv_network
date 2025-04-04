import numpy as np
from models.base_layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, W_init=None, b_init=None):
        np.random.seed(42)
        self.name = 'Dense'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.preactivation = None
        if W_init is None or b_init is None:
            self.W = np.random.random((input_dim, output_dim))
            self.b = np.zeros(output_dim, 'float32')
        else:
            self.W = W_init
            self.b = b_init

    def forward(self, input_data):
        self.preactivation = input_data
        out = np.empty((input_data.shape[0], self.output_dim))
        for b in range(input_data.shape[0]):
            out[b] = input_data[b] @ self.W + self.b
        return out

    def grad_x(self, input_data=None):
        if input_data is None:
            input_data = self.preactivation
        J = np.empty((input_data.shape[0], self.output_dim, self.input_dim))
        for b in range(input_data.shape[0]):
            J[b] = self.W.T
        return  J

    def grad_b(self, input_data):
        self.forward(input_data)
        batch_size = input_data.shape[0]
        J = np.empty((batch_size, self.output_dim, self.output_dim))
        for b in range(batch_size):
            J[b] = np.eye(self.output_dim)
        return J

    def grad_W(self, input_data):
        self.forward(input_data)
        batch_size = input_data.shape[0]
        J = np.empty((batch_size, self.output_dim, self.output_dim * self.input_dim))
        for b in range(batch_size):
            for i in range(self.output_dim):
                x_index = 0
                for j in range(self.output_dim * self.input_dim):
                    if (j % self.output_dim) == i:
                        J[b, i, j] = input_data[b, x_index]
                        x_index += 1
                    else:
                        J[b, i, j] = 0
        return J

    def update_W(self, grad, learning_rate):
        self.W -= learning_rate * np.mean(grad, axis=0).reshape(self.W.shape)

    def update_b(self, grad, learning_rate):
        self.b -= learning_rate * np.mean(grad, axis=0)

    def update_param(self, params_grad, learning_rate):
        self.update_W(params_grad[0], learning_rate)
        self.update_b(params_grad[1], learning_rate)

    def grad_param(self, input_data):
        return [self.grad_W(input_data), self.grad_b(input_data)]