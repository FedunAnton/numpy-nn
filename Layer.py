
class Layer:

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        return grad_output
