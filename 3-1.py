import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0,1,(n_inputs,n_neurons))
        self.bias = np.random.uniform(0,1,(1,n_neurons))

    def forword(self, inputs):
        return np.dot(inputs, np.array(self.weights)) + self.bias


nnfs.init()

X, y = spiral_data(samples=100, classes=2)
Layer1 = Layer_Dense(2,5)
Layer2 = Layer_Dense(5,2)
Layer2.forword(Layer1.forword(X))
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
plt.show()


