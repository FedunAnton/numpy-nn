import numpy as np
from Layer import Layer


class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad


class Tanh(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        tanh_forward = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        return tanh_forward

    def backward(self, input, grad_output):
        tanh_grad = 1 - self.forward(input) ** 2
        return grad_output * tanh_grad


class Sine(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        pass

    def forward(self, input):
        sine_forward = 0.9*np.sin(input)
        return sine_forward

    def backward(self, input, grad_output):
        sine_grad = 0.9*np.cos(input)
        return grad_output * sine_grad


class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        sigmoid_forward = 1 / (1 + np.exp(-input))
        return sigmoid_forward

    def backward(self, input, grad_output):
        sigmoid_grad = self.forward(input) * (1 - self.forward(input))
        return grad_output * sigmoid_grad

