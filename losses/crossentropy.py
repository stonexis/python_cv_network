
class CrossEntropy(object):
    def __init__(self, eps=0.00001):
        self.name = 'CrossEntropy'
        self.eps = eps

    def forward(self, input_data, labels):
        pass

    def calculate_loss(self, input_data, labels):
        return self.forward(input_data, labels)

    def grad_x(self, input_data, labels):
        pass