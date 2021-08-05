import numpy as np

from Function import Function


def binary_class_accuracy(predicted, target):
    """
    Applies a threshold for computing classification accuracy (correct classification rate).
    If the difference in absolute value between predicted - target is less than a specified threshold it considers it
    correctly classified (returns 1). Else returns 0
    The threshold is 0.3
    """
    predicted = predicted[0]
    target = target[0]
    if np.abs(predicted - target) < 0.3:
        return [1]
    return [0]


def euclidean_loss(predicted, target):
    """
    Computes the euclidean error between the target vector and the output predicted by the net
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth for each of n examples
    :return: error in terms of euclidean error
    """
    return np.linalg.norm(np.subtract(predicted, target))


BinClassAcc = Function(binary_class_accuracy, 'bin_class_acc')
Euclidean = Function(euclidean_loss, 'euclidean')
metrics = {
    'bin_class_acc': BinClassAcc,
    'euclidean': Euclidean
}
