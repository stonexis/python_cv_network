import numpy as np


class CrossEntropy(object):
    def __init__(self, eps=1e-5):
        self.name = 'CrossEntropy'
        self.eps = eps
        self.loss_value = None

    def forward(self, input_data, labels):
        logits = np.log(input_data + self.eps)
        loss = -np.sum(labels * logits, axis=1)
        self.loss_value = np.mean(loss)
        return self.loss_value

    def calculate_loss(self, input_data, labels):
        return self.forward(input_data, labels)

    def grad_x(self, input_data, labels):
        return -np.mean(labels / (input_data + self.eps), axis=0, keepdims=True)