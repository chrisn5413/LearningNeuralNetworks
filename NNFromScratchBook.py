# Simple neural network


# Input > layer1 > activation1 > layer2 > activation2 > loss

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# import matplotlib.pyplot as plt

nnfs.init()

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# RELU activation, forces node outputs to be 0 or positive only
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

# Softmax activation, gives meaning to individual values/features within each sample/feature set
#   by making each value a probability with respect to the sum of all values (between 0 and 1).
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities.
        # Overflow prevention: Subtract the largest value to limit the scale of e^x.
        #   x-x = 0, -->  e^0 = 1.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalizing gives a baseline that allows all samples to be relevant/comparable to each other
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Create dataset
X, y = spiral_data(100, 3)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# contains multiple stages of making a neural network from complex to simple
# Just choose one to run
def main():
    two_dense_two_active_loss()
    # two_dense_two_active()
    # one_dense_one_active()
    # one_dense()
    # one_dense_classless()


# Two layer dense
# Rectified Linear Units (ReLU) activation for Dense1
# Softmax activation for Dense2
# Loss pass for final output
def two_dense_two_active_loss():
    # Create Dense layer with 2 input features and 3 output values
    # Create ReLU activation (used with Dense Layer)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    # Create Dense layer with 2 input features and 3 output values
    # Create ReLU activation (used with Dense Layer)
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Takes output from previous layer and runs it through ReLU activation
    activation1.forward(dense1.output)

    # Perform a forward pass of our training data through this layer
    dense2.forward(activation1.output)

    # Takes output from previous layer and runs it through ReLU activation
    activation2.forward(dense2.output)

    # Output
    print(activation2.output[:5])

# Two layer dense
# Rectified Linear Units (ReLU) activation for Dense1
# Softmax activation for Dense2
def two_dense_two_active():
    # Create Dense layer with 2 input features and 3 output values
    # Create ReLU activation (used with Dense Layer)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    # Create Dense layer with 2 input features and 3 output values
    # Create ReLU activation (used with Dense Layer)
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Takes output from previous layer and runs it through ReLU activation
    activation1.forward(dense1.output)

    # Perform a forward pass of our training data through this layer
    dense2.forward(activation1.output)

    # Takes output from previous layer and runs it through ReLU activation
    activation2.forward(dense2.output)

    # Output
    print(activation2.output[:5])

# One layer dense
# Rectified Linear Units as activation for Dense1
def one_dense_one_active():
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)

    # Create ReLU activation (used with Dense Layer)
    activation1 = Activation_ReLU()

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Takes output from previous layer and runs it through ReLU activation
    activation1.forward(dense1.output)

    # Output
    print(activation1.output[:5])

# One layer dense
def one_dense():
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Output
    print(dense1.output[:5])

# Simple one layer dense with no classes
def one_dense_classless():
    inputs = [[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2.0, 3.0, 0.5]

    weights2 = [[0.1, -0.14, 0.5],
                [-0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]]
    biases2 = [-1, 2, -0.5]

    layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
    layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
    print(layer2_outputs)


if __name__ == '__main__':
    main()
