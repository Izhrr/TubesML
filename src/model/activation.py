import numpy as np

class Activations:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return (2 / (np.exp(x) + np.exp(-x)))**2 

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        s = Activations.softmax(x)
        return s * (1 - s)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def swish(x):
        return x * Activations.sigmoid(x)

    @staticmethod
    def swish_derivative(x):
        # f(x) = x * sigmoid(x)
        # f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        s = Activations.sigmoid(x)
        return s + x * s * (1 - s)