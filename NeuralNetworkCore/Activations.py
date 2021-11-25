import numpy as np

from NeuralNetworkCore.Function import DerivableFunction


def identity_function(x):
    """ identity (linear) activation function """
    return x


def identity_function_derivative(x):
    """ Computes the derivative of the identity function """
    return 1.


def relu_function(x):
    """ Computes the ReLU activation function """
    return np.maximum(x, 0)


def relu_function_derivative(x):
    """ Computes the derivative of the ReLU activation function """
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def leaky_relu_function(x):
    """ Computes the leaky ReLU activation function """
    return [i if i >= 0 else 0.01 * i for i in x]


def leaky_relu_function_derivative(x):
    """ Computes the derivative of the leaky ReLU activation function """
    x[x > 0] = 1.
    x[x <= 0] = 0.01
    return x


def sigmoid_function(x):
    """ Computes the sigmoid activation function """
    ones = [1.] * len(x)
    return np.divide(ones, np.add(ones, np.exp(-x)))


def sigmoid_function_derivative(x):
    """ Computes the derivative of the sigmoid activation function """
    return np.multiply(
        sigmoid_function(x),
        np.subtract([1.] * len(x), sigmoid_function(x))
    )


def tanh_function(x):
    """ Computes the hyperbolic tangent function (TanH) """
    return np.tanh(x)


def tanh_function_derivative(x):
    """ Computes the derivative of the hyperbolic tangent function (TanH) """
    return np.subtract(
        [1.] * len(x),
        np.power(np.tanh(x), 2)
    )

def softmax_function(x):
    """Computes the probability of a multi-class problem"""
    out = np.exp(x)
    return out/np.sum(out)


# activation functions
Identity = DerivableFunction(identity_function, identity_function_derivative, 'identity')
ReLU = DerivableFunction(relu_function, relu_function_derivative, 'ReLU')
LeakyReLU = DerivableFunction(leaky_relu_function, leaky_relu_function_derivative, 'LeakyReLU')
Sigmoid = DerivableFunction(sigmoid_function, sigmoid_function_derivative, 'Sigmoid')
Tanh = DerivableFunction(tanh_function, tanh_function_derivative, 'Tanh')
activation_functions = {
    'identity': Identity,
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'linear': Identity
}
