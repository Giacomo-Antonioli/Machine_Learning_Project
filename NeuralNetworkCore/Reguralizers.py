import numpy as np

from NeuralNetworkCore.Function import DerivableFunction


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


class EarlyStopping:

    def __init__(self, monitor='loss', mode='growth', patience=3, tolerance=1e-2):
        self.__possible_modes = ['growth', 'invariant', 'absolute_growth']
        self.__possible_monitors = ['loss', 'val']
        self.__tolerance = tolerance
        self.__patience = patience
        self.__validation_window = []
        self.__window_counter = 0
        if monitor in self.__possible_monitors:
            self.__monitor = monitor
        else:
            raise AttributeError("Invalid Monitor value")
        if mode in self.__possible_modes:
            self.__mode = mode
        else:
            raise AttributeError("Invalid Mode value")

    @property
    def monitor(self):
        return self.__monitor

    @monitor.setter
    def monitor(self, monitor):
        self.__monitor = monitor

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        self.__mode = mode

    def _push(self, new_val_score):
        if not self.__window_counter == self.__patience + 1:
            self.__validation_window.append(new_val_score)
            self.__window_counter += 1
        else:
            self.__validation_window = self.__validation_window[1:]
            self.__validation_window.append(new_val_score)

    def print_window(self):
        print(self.__validation_window)

    def _check_stopping(self):
        first = self.__validation_window[0]
        counter = 0
        for x in self.__validation_window[1:]:
            if self.__mode == 'absolute_growth':
                if x > first:
                    return True
            if self.__mode == 'growth':
                if x > first:
                    counter += 1
            if self.__mode == 'invariant':
                if np.abs(np.subtract(x, first)) < self.__tolerance:
                    counter += 1
        if counter == self.__patience:
            return True
        return False

    def apply_stop(self, val_score):
        self._push(val_score)
        if self.__window_counter == self.__patience + 1:
            return self._check_stopping()
        else:
            return False
