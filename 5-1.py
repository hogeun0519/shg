import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from unicodedata import category

nnfs.init()

class  oss_CategoricalCrossentropy:
    def forward(self,predictions, targets):
        predictions = np.clip(predictions,1e-7, 1- 1e-7)

        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)),targets]
        else:
            correct_confidences = np.sum(predictions * targets, axis=1, keepdims=True)

        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

    class Layer_Dense:
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))

        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases

    class Activation_ReLU:
        def forward(self, inputs):
            self.outputs = np.maximum(0, inputs)

    class Activation_Softmax:
        def forward(self, inputs):
            self.output = np.exp(inputs)

    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 3)  # 입력 레이어 (2 -> 3)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3, 3)  # 출력 레이어 (3 -> 3)
    activation2 = Activation_Softmax()


    dense1.forward(X)
    dense2.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)


    #Example
    # 가정: NN 맨마지막 Softmax activation function

    softmax_outputs = np.array([
        [0.7,0.1,0.2],
        [0.1,0.5,0.4],
        [0.2,0.2,0.6]
    ])
    targets = np.array([0,1,2])



loss = CrossentropyLoss.forward(activation2.output)
print(loss)