import numpy as np
inputs = [[1.5, 2.5, 3.5, 2.5],
          [1.0, 4.0, -2.0, 2.0],
          [-1.5, 2.5, 3.0, -0.8]]
weights = [[0.2, 0.7, -0.8, 1.5],
          [0.5, -0.98, 0.33, -0.6],
          [-0.5, -0.25, 0.13, 0.89]]

bias = [2.0, 3.0, 0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) + bias

print(layers_outputs)