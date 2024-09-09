import numpy as np
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]

bias = [2.0, 3.0, 0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) + bias

print(layers_outputs)

weights_2 = [[0.3, 0.5, -0.7]]

bias_2 = [2.5]

layers_outputs_2 = np.dot(layers_outputs, np.array(weights_2).T) + bias_2

print(layers_outputs_2)