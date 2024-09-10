import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons, initialize_method='uniform'):
        self.input_dim = n_inputs
        self.output_dim = n_neurons
        self.initialize_method = initialize_method

        self.weights = self.initialize_weights()
        self.bias = np.random.uniform(0, 1, (1, n_inputs))

    def initialize_weights(self):
        if self.initialize_method == 'uniform':
            return np.random.uniform(0, 1, (self.n_inputs, self.n_neurons))

        elif self.initialize_method == 'xavier':

            limit = np.sqrt(6 / (self.input_dim + self.output_dim))
            return np.random.uniform(-limit, limit, (self.n_inputs, self.n_neurons))

        elif self.initialize_method == 'he':
            # He 초기화
            stddev = np.sqrt(2 / self.input_dim)
            return np.random.normal(0, stddev, (self.n_inputs, self.n_neurons))

        else:
            raise ValueError(f"Unknown initialization method: {self.initialize_method}")

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

        self.output = np.maximum(0, self.output)

        return self.output