
import pytest
import numpy as np
from models.dense import DenseLayer

class TestDenseLayer:
    @pytest.fixture
    def dense_layer(self):
        """Fixture to create a DenseLayer for testing."""
        np.random.seed(42)  # Set seed for reproducibility
        return DenseLayer(input_dim=3, output_dim=2)


    def test_forward_shape(self, dense_layer):
        """Test that the forward pass produces output of the correct shape."""
        inputs = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])  # Single input sample with 3 features
        output = dense_layer.forward(inputs)
        assert output.shape == (2, 2), "Output shape should match (batch_size, output_size)"

    def test_forward_no_activation(self, dense_layer):
        """Test forward pass when no activation function is used."""
        inputs = np.array([[1.0, 2.0, 3.0]])
        expected_output = np.dot(inputs, dense_layer.W) + dense_layer.b

        output = dense_layer.forward(inputs)
        assert np.allclose(output, expected_output,
                           atol=1e-5), "Output should match linear transformation without activation"
