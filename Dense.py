import numpy as np
from Layer import Layer
np.random.seed(24)


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate):

        self.learning_rate = learning_rate
        self.weights1 = np.random.normal(size=(input_units, output_units))
        self.weights = np.random.randn((input_units * output_units)).reshape((input_units, output_units))\
                       / np.sqrt(output_units+input_units)
        self.biases = np.zeros(output_units)

    def forward(self, input):
        forward_prop = input.dot(self.weights)
        forward_prop += self.biases

        return forward_prop

    def backward(self, input, grad_output):

        grad_input = grad_output.dot(self.weights.T)
        self.weights = self.weights - self.learning_rate * input.T.dot(grad_output)
        self.biases = self.biases - self.learning_rate * grad_output.mean(axis=0) * input.shape[0]
        return grad_input
