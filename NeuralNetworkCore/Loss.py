import numpy as np
from math import log2
from math import log
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
    # exponent 2 in the derive becomes a multiplying constant and simplifies itself with the denominator of the func
    return np.subtract(predicted, target)


def binary_cross_entropy(predicted,target):
    sum=0
    for i in range(len(target)):
        sum+=target[i]*log(predicted[i])+(1-predicted[i])*log(1-predicted[i])
    return - sum/len(target)

def binary_cross_entropy_derivative(predicted,target):
    sum=0
    for i in range(len(target)):
        sum+=target[i]-predicted[i]
    return - sum/len(target)

# calculate categorical cross entropy
def categorical_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            sum_score += actual[i][j] * log(1e-15 + predicted[i][j])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score



# losses
SquaredLoss = DerivableFunction(squared_loss, squared_loss_derivative, 'squared')
losses = {
    'squared': SquaredLoss,
}
