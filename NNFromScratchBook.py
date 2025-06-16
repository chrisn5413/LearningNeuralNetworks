# Simple neural network


'''
Activations 1 -> N-1 are typically ReLU
Activation N is typically softmax
Input > layer1 > activation1 ... layerN > activationN > loss 
'''

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


# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# Categorical Cross Entropy uses two probability distributions, final output probability and the one-hot encoding
# One-hot encoding is a vector/list of classes with only one class active (0 is unactive, 1 is active) e.g. [0,1,0,0]
# To calculate, take the natural log of each output probability, multiply by corresponding one-hot, sum, then negate
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_target):
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        # values are indices for each sample: [0, 1, 1] means sample 1 index 0, sample 2 index 1, sample 3 index 1
        # Numpy arrays can be indexed given arrays: range -> [0, 1, 2], y_target -> [0, 1, 1], access [0,0], [1,1], [2,1]
        if len(y_target.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_target]

        # 2d target means each sample is one-hot encoded
        # [[1, 0, 0], [0, 1, 0], [0, 1, 0]], sample 1 index 0, sample 2 index 1, sample 3 index 1
        # by specifying axis 1, it sums the values along axis 1 for each row (all values per row)
        elif len(y_target.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_target, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)


# Create dataset
X, y = spiral_data(100, 3)
'''plt.scatter(X[:, 0], X[:, 1])
plt.show()'''


# contains multiple stages of making a neural network from complex to simple
# Just choose one to run
def main():
    neural_network()
    # two_dense_two_active_loss_accuracy()
    # two_dense_two_active()
    # one_dense_one_active()
    # one_dense()
    # one_dense_classless()


def neural_network():
    # Create model
    dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
    activation2 = Activation_Softmax()

    # Create loss function
    loss_function = Loss_CategoricalCrossEntropy()
    # Helper variables
    lowest_loss = 9999999  # some initial value
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    for iteration in range(10000):
        # Update weights with some small random values
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Perform a forward pass through activation function
        # it takes the output of second dense layer here and returns loss
        loss = loss_function.calculate(activation2.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        # If loss is smaller - print and save weights and biases aside
        if loss < lowest_loss:
            print('New set of weights found, iteration:', iteration,
                  'loss:', loss, 'acc:', accuracy)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss

        # Revert weights and biases
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()


# Two layer dense
# Rectified Linear Units (ReLU) activation for Dense1
# Softmax activation for Dense2
# Loss (error) calculation
# Accuracy (correct guesses) calculation
def two_dense_two_active_loss_accuracy():
    # Create Dense layer with 2 input features and 3 output values
    # Create ReLU activation (used with Dense Layer)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    # Create Dense layer with 3 input features and 3 output values
    # Create Softmax activation (used with final Dense Layer)
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    # Create loss function to calculate total mean loss for dataset
    loss_function = Loss_CategoricalCrossEntropy()

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Takes output from previous layer and runs it through ReLU activation
    activation1.forward(dense1.output)

    # Perform a forward pass of our training data through this layer
    dense2.forward(activation1.output)

    # Takes output from final Dense layer and runs it through Softmax activation
    activation2.forward(dense2.output)

    # Output
    print(activation2.output[:5])

    # Perform a forward pass through loss function
    loss = loss_function.calculate(activation2.output, y)

    print(f"loss: {loss}")

    class_targets = y
    # Converts one-hot 2d arrays into singular array
    if class_targets.shape == 2:
        class_targets = np.argmax(class_targets, axis=1)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == class_targets)

    print(f"accuracy: {accuracy}")

# Two layer dense
# Rectified Linear Units (ReLU) activation for Dense1
# Softmax activation for Dense2
def two_dense_two_active():
    # Create Dense layer with 2 input features and 3 output values
    # Create ReLU activation (used with Dense Layer)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    # Create Dense layer with 3 input features and 3 output values
    # Create Softmax activation (used with final Dense Layer)
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Takes output from previous layer and runs it through ReLU activation
    activation1.forward(dense1.output)

    # Perform a forward pass of our training data through this layer
    dense2.forward(activation1.output)

    # Takes output from previous layer and runs it through Softmax activation
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
