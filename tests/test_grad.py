import pytest
import inspect
import numpy as np
import models


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
