import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Network(object):
    def __init__(self, layers, loss=None):
        self.name = 'Network'
        self.layers = layers
        self.loss = loss

    def forward(self, input_data):
        return self.predict(input_data)

    def grad_x(self, input_data, labels):
        self.calculate_loss(input_data, labels)
        grad_L = self.loss.grad_x()
        jacobian_chain_old = np.repeat(grad_L, repeats=self.layers[-1].grad_x().shape[0], axis=0)
        for layer in reversed(self.layers):
            jacobian_chain_new = np.empty((jacobian_chain_old.shape[0], layer.grad_x().shape[2]))
            for b in range(jacobian_chain_old.shape[0]):
                jacobian_chain_new[b] = jacobian_chain_old[b] @ layer.grad_x()[b]
            jacobian_chain_old = jacobian_chain_new
        return jacobian_chain_old.mean(axis=0)

    def grad_param(self, input_data, labels):
        pass

    def update(self, grad_list, learning_rate):
        pass

    def predict(self, input_data):
        current_input = input_data
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def calculate_loss(self, input_data, labels):
        return self.loss.forward(self.predict(input_data), labels)

    def train_step(self, input_data, labels, learning_rate=0.001):
        grad_list = self.grad_param(input_data, labels)
        self.update(grad_list, learning_rate)

    def fit(self, trainX, trainY, validation_split=0.25,
            batch_size=1, nb_epoch=1, learning_rate=0.01):

        train_x, val_x, train_y, val_y = train_test_split(trainX, trainY,
                                                          test_size=validation_split,
                                                          random_state=42)
        for epoch in range(nb_epoch):
            # train one epoch
            for i in tqdm(range(int(len(train_x) / batch_size))):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                self.train_step(batch_x, batch_y, learning_rate)
            # validate
            val_accuracy = self.evaluate(val_x, val_y)
            print('%d epoch: val %.2f' % (epoch + 1, val_accuracy))

    def evaluate(self, testX, testY):
        y_pred = np.argmax(self.predict(testX), axis=1)
        y_true = np.argmax(testY, axis=1)
        val_accuracy = np.sum((y_pred == y_true)) / (len(y_true))
        return val_accuracy