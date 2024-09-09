import numpy as np
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8],
          [2.0, 3.5, 4.0, 1.7],
          [-3.0, 4.5, 2.7, -3.3]]
weights = [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]

bias = [2.0, 3.0, 0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) + bias

print(layers_outputs)