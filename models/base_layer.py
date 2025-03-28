class Layer(object):
    def __init__(self):
        self.name = 'Layer'

    def forward(self, input_data):
        pass

    def backward(self, input_data):
        return [self.grad_x(input_data), self.grad_param(input_data)]

    def grad_x(self, input_data):
        pass

    def grad_param(self, input_data):
        return []

    def update_param(self, grads, learning_rate):
        pass