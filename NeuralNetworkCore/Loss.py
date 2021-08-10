import numpy as np

from NeuralNetworkCore.Function import DerivableFunction


def squared_loss(predicted, target):
    """
    Computes the squared error between the target vector and the output predicted by the net
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth for each of n examples
    :return: loss in terms of squared error
    """
    return 0.5 * np.square(np.subtract(target, predicted))  # "0.5" is to make the gradient simpler


def squared_loss_derivative(predicted, target):
    """
    Computes the derivative of the squared error between the target vector and the output predicted by the net
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth for each of n examples
    :return: derivative of the squared error
    """
    # exponent 2 in the deriv becomes a multiplying constant and simplifies itself with the denominator of the func
    return np.subtract(predicted, target)


# losses
SquaredLoss = DerivableFunction(squared_loss, squared_loss_derivative, 'squared')
losses = {
    'squared': SquaredLoss,
}
