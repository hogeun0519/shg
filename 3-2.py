import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0,1,(n_inputs,n_neurons))
        self.bias = np.random.uniform(0,1,(1,n_neurons))

    def forword(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

        self.output = np.maximum(0, self.output)
        return self.output