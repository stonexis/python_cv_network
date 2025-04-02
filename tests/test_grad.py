import pytest
import inspect
import numpy as np
import models
import losses
from losses import CrossEntropy

LAYER_PARAMS = {
    "DenseLayer": {"input_dim": 3, "output_dim": 4},
    "ReLU": {},
    "Softmax": {},
}
IGNORED_LAYERS = {
    "Network",
    "Layer",
    "FlattenLayer",
    "MaxPooling",
}

@pytest.fixture
def layers():
    layers = []
    for name in models.__all__:
        if name in IGNORED_LAYERS:
            continue
        cls = getattr(models, name)

        if inspect.isclass(cls) and hasattr(cls, "grad_x"):
            print(f"Adding layer: {name}")
            params = LAYER_PARAMS.get(name, {})
            instance = cls(**params)
            layers.append(instance)
    return layers

@pytest.fixture
def net():
    return models.Network([models.DenseLayer(3, 10), models.ReLU(), models.DenseLayer(10, 3), models.Softmax()], loss=losses.CrossEntropy())

def test_grad_shape(layers):
    x = np.random.randn(4, 3)
    for layer in layers:
        y = layer.forward(x)
        jacobian = layer.grad_x(x)
        batch_size = x.shape[0]
        out_size = y.shape[1]
        in_size = x.shape[1]
        assert jacobian.shape == (batch_size, out_size, in_size)

def test_grad(layers):
    x = np.random.randn(4, 3)
    eps = 1e-5
    for layer in layers:
        y = layer.forward(x)
        jacobian = layer.grad_x(x)
        batch_size = x.shape[0]
        out_size = y.shape[1]
        in_size = x.shape[1]
        numeric_jac = np.empty((batch_size, out_size, in_size))
        E = np.zeros_like(x)
        for b in range(batch_size):
            for i in range(in_size):
                E[b, i] = eps
                for j in range(out_size):
                    x_perturbed = (x[b] + E[b])[np.newaxis, :]
                    y_plus = layer.forward(x_perturbed)[0, j]

                    x_perturbed = (x[b] - E[b])[np.newaxis, :]
                    y_minus = layer.forward(x_perturbed)[0, j]

                    numeric_jac[b, j, i] = (y_plus - y_minus) / (2 * eps)
                E[b, i] = 0
        if not np.allclose(jacobian, numeric_jac, atol=eps):
            diff = np.abs(jacobian - numeric_jac)
            max_diff = np.max(diff)
            print(f"Error in layer: {layer.__class__.__name__}")
            print(f"Max diff: {max_diff:.6e}")
            print(f"Analytical grad:\n{jacobian}")
            print(f"Numerical grad:\n{numeric_jac}")
        assert np.allclose(jacobian, numeric_jac, atol=eps)

def numerical_diff_net(net, x, labels):
    eps = 1e-5
    right_answer = []
    for i in range(len(x[0])):
        delta = np.zeros(len(x[0]))
        delta[i] = eps
        diff = (net.calculate_loss(x + delta, labels) - net.calculate_loss(x-delta, labels)) / (2*eps)
        right_answer.append(diff)
    return np.array(right_answer).T

def test_net(net):
    x = np.array([[1, 2, 3], [2, 3, 4]])
    labels = np.array([[0.3, 0.2, 0.5], [0.3, 0.2, 0.5]])
    num_grad = numerical_diff_net(net, x, labels)
    grad = net.grad_x(x, labels)
    if np.allclose(grad, num_grad, atol=1e-2):
        print('Test PASSED')
    else:
        print('Something went wrong!')
        print('Numerical grad is')
        print(num_grad)
        print('Your gradiend is ')
        print(grad)
    assert np.allclose(grad, num_grad, atol=1e-2)


def test_loss(net=CrossEntropy()):
    x = np.array([[1, 2, 3], [2, 3, 4]])
    labels = np.array([[0.3, 0.2, 0.5], [0.3, 0.2, 0.5]])
    num_grad = numerical_diff_net(net, x, labels)
    grad = net.grad_x(x, labels)
    if np.allclose(grad, num_grad, atol=1e-2):
        print('Test PASSED')
    else:
        print('Something went wrong!')
        print('Numerical grad is')
        print(num_grad)
        print('Your gradiend is ')
        print(grad)
    assert np.allclose(grad, num_grad, atol=1e-2)

def numerical_grad_b(input_size, output_size, b, W, x):
    eps = 1e-5
    right_answer = []
    for i in range(len(b)):
        delta = np.zeros(b.shape)
        delta[i] = eps
        dense1 = models.DenseLayer(input_size, output_size, W_init=W, b_init=b+delta)
        dense2 = models.DenseLayer(input_size, output_size, W_init=W, b_init=b-delta)
        diff = (dense1.forward(x) - dense2.forward(x)) / (2*eps)
        right_answer.append(diff.T)
    return np.array(right_answer).T

def test_grad_b():
    input_size = 3
    output_size = 4
    W_init = np.random.random((input_size, output_size))
    b_init = np.random.random((output_size,))
    x = np.random.random((2, input_size))

    dense = models.DenseLayer(input_size, output_size, W_init, b_init)
    grad = dense.grad_b(x)

    num_grad = numerical_grad_b(input_size, output_size, b_init, W_init, x)
    if np.allclose(grad, num_grad, atol=1e-2):
        print('Test PASSED')
    else:
        print('Something went wrong!')
        print('Numerical grad is')
        print(num_grad)
        print('Your gradiend is ')
        print(grad)
    assert np.allclose(grad, num_grad, atol=1e-2)

def numerical_grad_W(input_size, output_size, b, W, x):
    eps = 1e-5
    right_answer = []
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            delta = np.zeros(W.shape)
            delta[i, j] = eps
            dense1 = models.DenseLayer(input_size, output_size, W_init=W+delta, b_init=b)
            dense2 = models.DenseLayer(input_size, output_size, W_init=W-delta, b_init=b)
            diff = (dense1.forward(x) - dense2.forward(x)) / (2*eps)
            right_answer.append(diff.T)
    return np.array(right_answer).T

def test_grad_W():
    input_size = 3
    output_size = 4
    W_init = np.random.random((input_size, output_size))
    b_init = np.random.random((4,))
    x = np.random.random((2, input_size))

    dense = models.DenseLayer(input_size, output_size, W_init, b_init)
    grad = dense.grad_W(x)

    num_grad = numerical_grad_W(input_size, output_size, b_init, W_init, x)
    if np.allclose(grad, num_grad, atol=1e-2):
        print('Test PASSED')
    else:
        print('Something went wrong!')
        print('Numerical grad is')
        print(num_grad)
        print('Your gradiend is ')
        print(grad)
    assert np.allclose(grad, num_grad, atol=1e-2)