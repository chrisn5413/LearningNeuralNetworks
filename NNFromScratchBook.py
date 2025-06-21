# Simple neural network


'''
Activations 1 -> N-1 are typically ReLU
Activation N is typically softmax
Input > layer1 > activation1 ... layerN > activationN > loss 
'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
# import matplotlib.pyplot as plt


nnfs.init()

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# RELU activation, forces node outputs to be 0 or positive only
# this function is effectively f(x) = x (or linear), but by cutting off negative values,
# nodes are able to more easily represent nonlinear functions without being a complex linear function itself
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # we'll make a copy first
        self.dinputs = dvalues.copy()

        # Zero gradient wherer input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation, gives meaning to individual values/features within each sample/feature set
#   by making each value a probability with respect to the sum of all values (between 0 and 1).
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities.
        # Overflow prevention: Subtract the largest value to limit the scale of e^x.
        #   x-x = 0, -->  e^0 = 1.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        # gives a baseline that allows all samples to be relevant/comparable to each other
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Categorical Cross Entropy uses two probability distributions, final output probability and the one-hot encoding
# One-hot encoding is a vector/list of classes with only one class active (0 is unactive, 1 is active) e.g. [0,1,0,0]
# To calculate, take the natural log of each output probability, multiply by corresponding one-hot, sum, then negate
class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        # Numbers of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        # values are indices for each sample: [0, 1, 1] means sample 1 index 0, sample 2 index 1, sample 3 index 1
        # Numpy arrays can be indexed given arrays: range -> [0, 1, 2], y_target -> [0, 1, 1], access [0,0], [1,1], [2,1]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # 2d shape means each sample is one-hot encoded
        # [[1, 0, 0], [0, 1, 0], [0, 1, 0]], sample 1 index 0, sample 2 index 1, sample 3 index 1
        # by specifying axis 1, it sums the values along axis 1 for each row (all values per row)
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of features in every sample
        # We'll use the first sample to count them
        features = len(dvalues[0])

        # If features are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(features)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossEntropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        # Result is a 1d vector with the index of the highest value
        # [[1,0,0], [0,0,1], [0,1,0]] => [0,2,1]
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient, 0->len(samples) array, y_true array
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Stochastic Gradient Descent, adjusts layer weights and biases by multiplying parameter gradients
# by the negated learning rate (default initialization is 1.0)
class Optimizer_SGD:
    
    # Initialize optimizer - set settings
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases



# Create dataset
X, y = spiral_data(samples=100, classes=3)
'''plt.scatter(X[:, 0], X[:, 1])
plt.show()'''


# contains multiple stages of making a neural network from complex to simple
# Just choose one to run
def main():
    single_hidden_layer_densely_connected_nn(X, y)
    # two_dense_two_active_with_backpropagation(X, y)
    # test_combined_softmax_loss_backward_method_vs_individual() # combined is faster and now the default
    # forward_and_backward_pass_of_one_neuron()
    # calculate_relu_gradient()
    # calculate_gradient_with_respect_to_bias()
    # calculate_gradient_with_respect_to_weight()
    # calculate_gradient_with_respect_to_input()
    # simple_one_neuron_with_backpropagation()

    # neural_network_random_weights_and_biases()    # cumulative of the below functions
    # two_dense_two_active_loss_accuracy()
    # two_dense_two_active()
    # one_dense_one_active()
    # one_dense()
    # one_dense_classless()



def single_hidden_layer_densely_connected_nn(X, y):
    # Uses the dataset in global
    # X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 64 output values
    dense1 = Layer_Dense(2, 64)

    # Create ReLU activation (used with Dense layer)
    activation1 = Activation_ReLU()

    # Create second Dense layer with 64 input features.
    # Takes output of previous layer and outputs 3 values
    dense2 = Layer_Dense(64, 3)

    # Craete Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

    # Create optimizer
    optimizer = Optimizer_SGD()

    # Train in loop
    for epoch in range(10001):

        # Forward pass sample data
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        # Print loss
        # print('loss: ', loss)

        # Calculate accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(predictions==y)
        accuracy = np.mean(predictions==y)

        # print('acc: ', accuracy)

        if not epoch % 100:
            print(f'epoch: {epoch}, ' + 
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)



# Single forward and backward pass of neural network with 1 hidden layer.
# Uses combined softmax/cross entropy calculation for forward/backward pass
def two_dense_two_active_with_backpropagation(X, y):
    # Uses the dataset in global
    # since y has a change of being reassigned: y = np.argmax(y, axis=1), 
    # this function wants explicit declaration of local variable, 
    # thus a parameter is used instead

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2,3)

    # Create ReLU activation (use with Dense Layer)
    activation1 = Activation_ReLU()

    # Create second Dense layer with 3 input features and 3 output values
    # IN: last layer output OUT: 3 output values
    dense2 = Layer_Dense(3,3)

    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

    # Perform a forward pass of our training data through first dense
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the first output of first dense layer
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes output of the first activation function
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer and returns loss
    loss = loss_activation.forward(dense2.output, y)

    # Let's see output of the first few samples:
    print('first 5 samples:\n', loss_activation.output[:5])

    # Print loss value
    print('loss:', loss)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    # Print accuracy
    print('acc:', accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    print('dense1.dweights:\n', dense1.dweights)
    print('dense1.dbiases:\n', dense1.dbiases)
    print('dense2.dweights:\n', dense2.dweights)
    print('dense2.dbiases:\n', dense2.dbiases)


# A test to show that the new combined method which was simplified
# with calculus results in the same answer (testing shows 7x faster)
def test_combined_softmax_loss_backward_method_vs_individual():
    # example softmax output (probability for each output)
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    
    class_targets = np.array([0, 1, 1])

    # Calculate using the combined method
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs

    # Calculating them separately
    activation = Activation_Softmax()
    activation.output = softmax_outputs
    loss = Loss_CategoricalCrossEntropy()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs

    print('Gradients: combined loss and activation:')
    print(dvalues1)
    print('Gradients: separate loss and activation:')
    print(dvalues2)


def forward_and_backward_pass_of_one_neuron():
    # Passed in gradient from the next layer
    # for the purpose of this example we're going to use
    # an array of incremental gradient values
    # dvalues = np.array([[1., 1., 1.],
    #                     [2., 2., 2.],
    #                     [3., 3., 3.]])        # we used drelu instead since we copied the output

    # We have 3 setts of inputs - samples
    inputs = np.array([[1, 2, 3, 2.5],
                        [2., 5., -1., 2],
                        [-1.5, 2.7, 3.3, -0.8]])

    # We have 3 sets of weights - one set for each neuron
    # we have 4 inputs, thus 4 weights
    # recall that we keep weights transposed
    weights = np.array([[0.2, 0.8, -0.5, 1],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T

    # One bias for each neuron
    # biases are the row vector with a shape (1, neurons)
    biases = np.array([[2, 3, 0.5]])

    # Forward pass
    layer_outputs = np.dot(inputs, weights) + biases    # Dense layer (each row has outputs for each neuron
    relu_outputs = np.maximum(0, layer_outputs)         # ReLU activation

    # Let's optimize and test back propagation here
    # ReLU activation - simulates derivative with respect to input values
    # from next layer passed to current layer during backpropagation
    drelu = relu_outputs.copy()
    drelu[layer_outputs <= 0] = 0


    # Dense layer
    # dinputs - multiply by weights
    dinputs = np.dot(drelu, weights.T)
    # dweights - multiply by inputs
    dweights = np.dot(inputs.T, drelu)
    # dbiases - sum values, do this over samples (first axis), use keepdims
    # since this by default will produce a plain list
    dbiases = np.sum(drelu, axis=0, keepdims=True)

    # Update parameters (first look on SGD-Stochastic Gradient Descent)
    weights += -0.001 * dweights
    biases += -0.001 * dbiases

    print(weights)
    print(biases)


# shows how to get the gradient derivative for the ReLU function
def calculate_relu_gradient():
    # Example layer output
    z = np.array([[1, 2, -3, -4],
                  [2, -7, -1, 3],
                  [-1, 2, 5, -1]])

    dvalues = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])

    # ReLU activation's derivative
    # drelu = np.zeros_like(z)
    # drelu[z > 0] = 1
    # print(drelu)

    # The chain rule
    # drelu *= dvalues
    # print(drelu)

    # shortens the above code by directly settings values in drelu to zero based on array z <= 0
    drelu = dvalues.copy()
    drelu[z < 0] = 0
    print(drelu)


# shows how to get the gradient derivative with respect to the bias
def calculate_gradient_with_respect_to_bias():
    # Passed in gradient from the next layer
    # the next layer provides a gradient for each neuron
    # for the purpose of this example we're going to use an array of incremental gradient values
    dvalues = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])

    # One bias for each neuron
    # biases are the row vector with a shape (1, neurons)
    biases = np.array([[2, 3, 0.5]])

    # dbiases - sum values, do this over samples (first axis), keep dims
    # since this by default will produce a plain list - we explained this in chapter 4
    dbiases = np.sum(dvalues, axis=0, keepdims=True)

    print(dbiases)


# shows how to get the gradient derivative with respect to the weight
def calculate_gradient_with_respect_to_weight():
    # Passed in gradient from the next layer
    # the next layer provides a gradient for each neuron
    # for the purpose of this example we're going to use an array of incremental gradient values
    dvalues = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])

    # We have 3 sets of inputs - samples
    inputs = np.array([[1, 2, 3, 2.5],
                        [2., 5., -1., 2],
                        [-1.5, 2.7, 3.3, -0.8]])

    # sum weights of given input
    # and multiply by the passed in gradient for this neuron
    # dweights is the derivative of the neuron function with respect to the weights
    dweights = np.dot(inputs.T, dvalues)

    print(dweights)


# shows how to get the gradient derivative with respect to the input
def calculate_gradient_with_respect_to_input():
    # Passed in gradient from the next layer in a normal pass

    # dvalues = np.array([[1., 1., 1.]])        # In this example we are using a vector of 1s

    # the next layer provides a gradient for each neuron
    # for the purpose of this example we're going to use an array of incremental gradient values
    dvalues = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])

    # We have 3 sets of weights - one set for each neuron
    # we have 4 inputs, thus 4 weights
    # recall that we keep weights transposed
    weights = np.array([[0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T

    # old code, what np.dot() does at a low level (the dot function is faster)
    # dx0 = sum(weights[0]) * dvalues[0]
    # dx1 = sum(weights[1]) * dvalues[0]
    # dx2 = sum(weights[2]) * dvalues[0]
    # dx3 = sum(weights[3]) * dvalues[0]
    # dinputs = np.array([dx0, dx1, dx2, dx3])

    # sums weights of given input
    # and multiply by the passed in gradient from this neuron
    # dinputs = np.dot(dvalues[0], weights.T)       # old code, from the single dvalues vector

    # dinputs is the derivative of the neuron function with respect to the input
    dinputs = np.dot(dvalues, weights.T)

    print(dinputs)


# This is an example of how to reduce the output of a single neuron.
# This is not something that would normally be done since usually you need to
# do this to minimize the loss. (This is only to show how it works and that it does
# reduce the final output value)
def simple_one_neuron_with_backpropagation():
    # Forward pass
    x = [1.0, -2.0, 3.0]  # input values
    w = [-3.0, -1.0, 2.0]  # weights
    b = 1.0  # bias

    # Multiplying inputs by weights (xw0, xw1, xw2: represent functions variables)
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]
    # print(xw0, xw1, xw2, b)

    # Adding weighted inputs and a bias
    z = xw0 + xw1 + xw2 + b
    # print(z)

    # ReLU activation function
    y = max(z,0)
    print("original y ",y)

    # Use the chain rule with derivatives for back propagation
    # y(z) = y(z(xw0,xw1,xw2,b)) = ReLU(sum(mul(x0,w0) + mul(x1,w1) + mul(x2,w2) + b), 0)

    # y'(z): if output == z, derivative of a variable is 1. if output == 0, derivative of constant is 0

    # d/dxw0 -> z(xw0,xw1,xw2,b):  xw0+xw1+xw2+b  =  1+0+0+0  =  1   (dsum_dxw0) partial derivative of sum with respect to h
    # d/dxw1 -> z(xw0,xw1,xw2,b):  xw0+xw1+xw2+b  =  0+1+0+0  =  1   (dsum_dxw1) partial derivative of sum with respect to i
    # d/dxw2 -> z(xw0,xw1,xw2,b):  xw0+xw1+xw2+b  =  0+0+1+0  =  1   (dsum_dxw2) partial derivative of sum with respect to j
    # d/db   -> z(xw0,xw1,xw2,b):  xw0+xw1+xw2+b  =  0+0+0+1  =  1   (dsum_db) partial derivative of sum with respect to b

    #   d/dx -> xw0(x0,w0): x0*w0  =  w0*1  =  w0    (dmul_dx0) partial derivative of multiply with respect to x
    #   d/dx -> xw1(x1,w1): x1*w1  =  w1*1  =  w1    (dmul_dx1) partial derivative of multiply with respect to x
    #   d/dx -> xw2(x2,w2): x2*w2  =  w2*1  =  w2    (dmul_dx2) partial derivative of multiply with respect to x
    #   d/dw -> xw0(x0,w0): x0*w0  =  x0*1  =  x0    (dmul_dw0) partial derivative of multiply with respect to w
    #   d/dw -> xw1(x1,w1): x1*w1  =  x1*1  =  x1    (dmul_dw1) partial derivative of multiply with respect to w
    #   d/dw -> xw2(x2,w2): x2*w2  =  x2*1  =  x2    (dmul_dw2) partial derivative of multiply with respect to w

    # back propagation
    # The derivative of ReLU and the chain rule
    dvalue = 1       # example assumption of next derivative (layer to the right, we are going left)
    drelu_dz = dvalue * (1. if z > 0 else 0.)     # z is 6 so relu_dz = 1
    # print(drelu_dz)

    # Partial derivatives of the multiplication, the chain rule
    dsum_dxw0 = 1
    dsum_dxw1 = 1
    dsum_dxw2 = 1
    dsum_db = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0
    drelu_dxw1 = drelu_dz * dsum_dxw1
    drelu_dxw2 = drelu_dz * dsum_dxw2
    drelu_db = drelu_dz * dsum_db
    # print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

    # Partial derivatives of the multiplication, the chain rule
    dmul_dx0 = w[0]
    dmul_dx1 = w[1]
    dmul_dx2 = w[2]
    dmul_dw0 = x[0]
    dmul_dw1 = x[1]
    dmul_dw2 = x[2]
    drelu_dx0 = drelu_dxw0 * dmul_dx0
    drelu_dx1 = drelu_dxw1 * dmul_dx1
    drelu_dx2 = drelu_dxw2 * dmul_dx2
    drelu_dw0 = drelu_dxw0 * dmul_dw0
    drelu_dw1 = drelu_dxw1 * dmul_dw1
    drelu_dw2 = drelu_dxw2 * dmul_dw2
    print("partial derivatives x0,w0,x1,w1,x2,w2:\n   ", drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

    # all partial derivatives of the single neuron
    dx = [drelu_dx0, drelu_dx1, drelu_dx2]  # gradients on inputs
    dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradients on weights
    db = drelu_db  #  gradient on bias (there is only 1 bias)

    print("original weight and bias:\n   ",w,b)

    w[0] += -0.001 * dw[0]
    w[1] += -0.001 * dw[1]
    w[2] += -0.001 * dw[2]
    b += -0.001 * db

    print("slightly tweeked weight and bias:\n   ", w, b)


    # New Forward pass with tweeked values
    # Multiplying inputs by weights (xw0, xw1, xw2: represent functions variables)
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]
    # print(xw0, xw1, xw2, b)

    # Adding
    z = xw0 + xw1 + xw2 + b

    # ReLU activation function
    y = max(z,0)
    print("new y ",y)



# below is the beginning of a neural network up till the end of the forward pass

# runs a simple neural network with an easy dataset (vertical data)
# to show why random weights/biases don't work well, comment out
# vertical data to use spiral data
def neural_network_random_weights_and_biases():
    # Comment the vertical data below to use spiral data
    # X, y = vertical_data(100, 3)

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