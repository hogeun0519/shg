import numpy as np
import nnfs
import matplotlib.pyplot as plt


from nnfs.datasets import spiral_data
nnfs.init()
class Activation_Softmax:

    def forward (self, inputs):
        # self.output = np.maximum(0, inputs)
        return np.maximum(0, inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0,1,(n_inputs, n_neurons))
        self.biases = np.zeros((1,n_neurons))
    def forward (self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

dense1 = Layer_Dense(1,8)
dense2 = Layer_Dense(8,8)
dense3 = Layer_Dense(8,1)

dense1.weights = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],])
dense1.biases = np.array([[0,0,0,0,0,0,0,0]])

dense2.weights = np.random.uniform(0,1,(3,3))
dense2.biases = np.zeros((1,8))

Activation3 = Activation_Softmax()

x = np.linspace(0,2*np.pi,100).reshape(-1,1)
y = np.sin(x)


plt.plot(x, y, label = "True Sine Wave", color='blue')
plt.plot(x, Activation3.forward(x), label = "NN Output", color='red')
plt.legend()
plt.title("Sine Wave Approximation using Neural Network")
plt.show()


output = Activation3.forward(dense1.forward(x))
print(output)
