import numpy as np

from Function import DerivableFunction


def lasso_l1(w, lambd):
    """
    Computes Lasso regularization (L1) on the nets' weights
    :param w: (list of matrices) the list of each layer's weights
    :param lambd: (float) regularization coefficient
    """
    return lambd * np.sum(np.abs(w))


def lasso_l1_deriv(w, lambd):
    """
    Computes the derivative of the Lasso regularization (L1)
    :param w: (list of matrices) the list of each layer's weights
    :param lambd: (float) regularization coefficient
    """
    res = np.zeros(w.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i][j] < 0:
                res[i][j] = -lambd
            elif w[i][j] > 0:
                res[i][j] = lambd
            else:
                res[i][j] = 0
    return res


def ridge_l2(w, lambd):
    """
    Computes Tikhonov regularization (L2) on the nets' weights
    :param w: (list of matrices) the list of each layer's weights
    :param lambd: (float) regularization coefficient
    """
    return lambd * np.sum(np.square(w))


def ridge_l2_deriv(w, lambd):
    """
    Computes the derivative of Tikhonov regularization (L1)
    :param w: (list of matrices) the list of each layer's weights
    :param lambd: (float) regularization coefficient
    """
    return 2 * lambd * w


l2_regularization = DerivableFunction(ridge_l2, ridge_l2_deriv, 'l2')
l1_regularization = DerivableFunction(lasso_l1, lasso_l1_deriv, 'l1')
regularizers = {
    'l2': l2_regularization,
    'l1': l1_regularization
}
