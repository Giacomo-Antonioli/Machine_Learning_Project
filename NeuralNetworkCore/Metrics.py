from math import pow

import numpy as np

from NeuralNetworkCore.Function import Function


def binary_class_accuracy(predicted, target):
    """
    Applies a threshold for computing classification accuracy (correct classification rate).
    If the difference in absolute value between predicted - target is less than a specified threshold it considers it
    correctly classified (returns 1). Else returns 0
    The threshold is 0.3
    """
    predicted = predicted[0]
    target = target[0]
    if np.abs(predicted - target) < 0.5:
        return [1]
    return [0]


def mean_euclidean_error(predicted, target):
    """
    Computes the euclidean error between the target vector and the output predicted by the net
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth for each of n examples
    :return: error in terms of euclidean error
    """
    return np.linalg.norm(np.subtract(predicted, target))


def accuracy(predicted, target):
    counter = 0
    for index in range(len(target)):
        if predicted[index] == target[index]:
            counter += 1
    if counter == 0:
        return counter
    else:
        return counter / len(target)


def true_false_positive(predicted, target):
    """
    The true positive rate is calculated as the number of true positives divided by the sum of the number of
    true positives and the number of false negatives.
    It describes how good the model is at predicting the positive class when the actual outcome is positive.
    :param predicted:
    :param target:
    :return: tpr, fpr
        WHERE
        tpr is the true positive rate
        fpr is the false positive rate
    """
    true_positive = np.equal(predicted, 1) & np.equal(target, 1)
    true_negative = np.equal(predicted, 0) & np.equal(target, 0)
    false_positive = np.equal(predicted, 1) & np.equal(target, 0)
    false_negative = np.equal(predicted, 0) & np.equal(target, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr


def mean_absolute_error(predicted, target):
    '''
    The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value
    of the absolute error loss or l1-norm loss.
    :param predicted:
    :param target:
    :return:
    '''
    return abs(np.subtract(predicted, target)) / len(target)


def squared_error(predicted, target):
    return pow(np.subtract(predicted, target), 2)


def mean_squared_error(predicted, target):
    '''
    The mean_squared_error function computes mean square error, a risk metric corresponding to
    the expected value of the squared (quadratic) error or loss.
    :param predicted:
    :param target:
    :return:
    '''
    return squared_error(predicted, target) / len(target)


def r2_score(predicted, target):
    '''
    The r2_score function computes the coefficient of determination, usually denoted as R².
    It represents the proportion of variance (of y) that has been explained by the independent variables in the model.
    It provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely
    to be predicted by the model, through the proportion of explained variance.
    As such variance is dataset dependent, R² may not be meaningfully comparable across different datasets.
    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
    A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score
    of 0.0.

    :param predicted:
    :param target:
    :return:
    '''
    sum = 0
    mean_target = np.mean(target)
    for index in len(range(target)):
        sum += pow(predicted[index] - mean_target, 2)

    return 1 - (squared_error(predicted, target) / mean_target)


BinClassAcc = Function('binary', binary_class_accuracy)
MEE = Function('mee', mean_euclidean_error)
Accuracy = Function('accuracy', accuracy)
MSE = Function('mse', mean_squared_error)
MAE = Function('mae', mean_absolute_error)
R2 = Function('r2', r2_score)
metrics = {
    'binary': BinClassAcc,
    'mee': MEE,
    'accuracy': Accuracy,
    'acc': Accuracy,
    'mse': MSE,
    'mae': MAE,
    'r2': R2
}
