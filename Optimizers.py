from tqdm import tqdm
import numpy as np
import math
from NeuralNetworkCore.Loss import losses
from NeuralNetworkCore.Metrics import metrics


class Optimizer:
    """
    class representing a generic gradient descend optimizer for a neural network model
    :param type : type of the optimizer
    :param lr : learning rate
    :param name : name of the optimizer
    :param loss_function : loss function applied
    :param metrics : metrics for the error
    :param stop_flag : flag for checking early stopping
    :param check_stop : early stopping criteria
    :param model : neural network model used for training
    :param train_datasets: training datasets
    :param train_labels: training data's label
    :param batch_size: size of batches
    :param pbar: dynamic bar visualizer
    :param gradient_network: gradient
    :param epoch_training_error: training error for each epoch
    :param epoch_training_error_metric: metric used to calculate each epoch training error
    """

    def __init__(self, learning_rate=0.1, name='', loss='', metric='', stop_flag=False):
        self.type = 'optimizer'
        self.lr = learning_rate
        self.name = name
        self.loss_function = loss
        self.metric = metric
        self.stop_flag = stop_flag
        self.check_stop = None
        self.values_dict = {
            'training_error': [],
            'training_metrics': [],
            'validation_error': [],
            'validation_metrics': []
        }
        self.model = ''
        self.train_dataset = ''
        self.train_labels = ''
        self.batch_size = 0
        self.pbar = ''
        self.gradient_network = ''
        self.epoch_training_error = ''
        self.epoch_training_error_metric = ''

    #region Optimezer properties
    @property
    def values_dict(self):
        return self.__values_dict

    @values_dict.setter
    def values_dict(self, new_value_dict):
        self.__values_dict = new_value_dict

    def set_values_dict_element(self, index, value):
        self.__values_dict[index].append(value)

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        self.__type = value

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, lr):
        if lr <= 0 or lr > 1:
            raise ValueError('learning_rate should be a value between 0 and 1, Got:{}'.format(lr))
        self.__lr = lr

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def loss_function(self):
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, loss_function):
        self.__loss_function = loss_function

    @property
    def metric(self):
        return self.__metric

    @metric.setter
    def metric(self, metric):
        self.__metric = metric

    @property
    def stop_flag(self):
        return self.__stop_flag

    @stop_flag.setter
    def stop_flag(self, stop=False):
        self.__stop_flag = stop

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def train_dataset(self):
        return self.__train_dataset

    @train_dataset.setter
    def train_dataset(self, train_dataset):
        self.__train_dataset = train_dataset

    @property
    def train_labels(self):
        return self.__train_labels

    @train_labels.setter
    def train_labels(self, train_labels):
        self.__train_labels = train_labels

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, size):
        self.__batch_size = size

    @property
    def pbar(self):
        return self.__pbar

    @pbar.setter
    def pbar(self, pbar):
        self.__pbar = pbar

    @property
    def check_stop(self):
        return self.__check_stop

    @check_stop.setter
    def check_stop(self, check_stop):
        self.__check_stop = check_stop

    @property
    def gradient_network(self):
        return self.__gradient_network

    @gradient_network.setter
    def gradient_network(self, gradient_network):
        self.__gradient_network = gradient_network

    @property
    def epoch_training_error(self):
        return self.__epoch_training_error

    @epoch_training_error.setter
    def epoch_training_error(self, epoch_training_error):
        self.__epoch_training_error = epoch_training_error

    @property
    def epoch_training_error_metric(self):
        return self.__epoch_training_error_metric

    @epoch_training_error_metric.setter
    def epoch_training_error_metric(self, epoch_training_error_metric):
        self.__epoch_training_error_metric = epoch_training_error_metric

    #endregion

    def check_instances(self, early_stopping=False, check_stop=None):
        # function to check consistency of variables

        # Checking if the early stopping parameter is set to true in order to stop the epoches loop based on the
        # validation curve
        if early_stopping is True and check_stop is None:
            raise AttributeError("CheckStop value is invalid")
        self.stop_flag = False
        self.check_stop = check_stop
        # Getting the loss function if this was a string or directly the function
        if isinstance(self.__model.loss, str):
            self.loss_function = losses[self.__model.loss]
        else:
            self.loss_function = self.__model.loss
        # Getting the metric function if this was a string or directly the function
        if isinstance(self.__model.metrics, str):
            self.metric = metrics[self.__model.metrics]
        else:
            self.metric = self.__model.metrics

    def check_batch(self, train_dataset, train_labels, batch_size=1):
        # checking the dimension of the batch division in order to divide the training in more batches and checking the
        # consistency of the subdivision in order to avoid runtime errors
        if batch_size is None:
            self.batch_size(1)
        elif batch_size == 'all':
            self.batch_size = len(train_dataset)
        else:
            self.batch_size = batch_size
        self.train_dataset = train_dataset[np.newaxis, :] if len(train_dataset.shape) < 2 else train_dataset
        self.train_labels = train_labels[np.newaxis, :] if len(train_labels.shape) < 2 else train_labels

    def set_batches(self, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        train_batch = self.train_dataset[start: end]
        targets_batch = self.train_labels[start: end]
        return train_batch, targets_batch

    def shuffle_data(self, train_dataset, train_labels):
        # shuffle the data
        indexes = list(range(len(train_dataset)))
        np.random.shuffle(indexes)
        self.train_dataset = train_dataset[indexes]
        self.train_labels = train_labels[indexes]

    def fit_with_gradient(self, input, target):
        # compute the propagation for each single input
        net_outputs = self.model.forward(net_input=input, training=True)

        # Sum the loss over all the elements in the batch
        self.epoch_training_error = np.add(self.epoch_training_error,
                                           self.__loss_function.function(predicted=net_outputs,
                                                                         target=target))
        # Sum the metric over all the elements in the batch
        self.epoch_training_error_metric = np.add(self.epoch_training_error_metric,
                                                  self.__metric.function(predicted=net_outputs,
                                                                         target=target))
        # compute the delta for the outmost layer
        dErr_dOut = self.loss_function.derive(predicted=net_outputs, target=target)
        # accumulate all the deltas for each layer of the batch computation
        gradient_network = self.model.propagate_back(dErr_dOut, self.gradient_network)

    def compute_training_error(self):
        self.epoch_training_error = np.sum(self.epoch_training_error) / len(self.epoch_training_error)
        current_loss_error = self.epoch_training_error / len(self.train_dataset)
        self.set_values_dict_element('training_error', current_loss_error)
        self.epoch_training_error_metric = np.sum(self.epoch_training_error_metric) / len(
            self.epoch_training_error_metric)
        self.set_values_dict_element('training_metrics', self.epoch_training_error_metric / len(self.train_dataset))
        return current_loss_error

    def init_epoch_training_error(self):
        self.epoch_training_error = np.zeros(self.model.layers[-1].n_units)
        self.epoch_training_error_metric = np.zeros(self.model.layers[-1].n_units)

    def validate(self, validation):
        val_x = validation[0][np.newaxis, :] if len(validation[0].shape) < 2 else validation[0]
        val_y = validation[1][np.newaxis, :] if len(validation[1].shape) < 2 else validation[1]
        epoch_val_error, epoch_val_metric = self.model.evaluate(validation_data=val_x, targets=val_y)
        current_val_error = epoch_val_error
        self.set_values_dict_element('validation_error', current_val_error)
        self.set_values_dict_element('validation_metrics', epoch_val_metric)
        return current_val_error

    def apply_stopping(self, current_loss_error, current_val_error):
        if self.check_stop.monitor == 'loss':
            self.stop_flag = self.check_stop.apply_stop(current_loss_error)
        elif self.check_stop.monitor == 'val':
            self.stop_flag = self.check_stop.apply_stop(current_val_error)

    def do_epochs(self, validation, epochs, shuffle, early_stopping, optimizer):
            current_val_error = 0
            epoch = 0
            self.pbar = tqdm(total=epochs)
            while epoch < epochs and not self.stop_flag:
                self.init_epoch_training_error()
                if self.batch_size != self.train_dataset.shape[0] and shuffle:
                    self.shuffle_data(self.train_dataset, self.train_labels)

                for batch_index in range(math.ceil(len(self.train_dataset) / self.batch_size)):
                    train_batch, targets_batch = self.set_batches(batch_index)
                    self.gradient_network = self.model.get_empty_struct()
                    for current_input, current_target in zip(train_batch, targets_batch):
                        self.fit_with_gradient(current_input, current_target)
                    optimizer(batch_index)
                if validation is not None:
                    current_val_error = self.validate(validation)

                current_loss_error = self.compute_training_error()
                if early_stopping:
                    self.apply_stopping(current_loss_error, current_val_error)
                epoch += 1
                self.pbar.update(1)
            self.pbar.close()


class StochasticGradientDescent(Optimizer):
    """
    Stochastic Gradient Descent
    """

    def __init__(self, learning_rate=0.1, momentum=0.1, nesterov=False):
        """
        Constructor
        :param learning_rate: Learning Rate parameter
        :param momentum:  Momentum parameter
        :param nesterov: Bool that indicates the usage of the nesterov momentum technique
        """
        super().__init__(learning_rate=learning_rate, name='sgd')
        self.momentum = momentum
        self.momentum_network = ''
        self.partial_momentum_network = ''
        self.nesterov = nesterov

    #region SGD properties
    @property
    def nesterov(self):
        return self.__nesterov

    @nesterov.setter
    def nesterov(self, nesterov):
        self.__nesterov = nesterov

    @property
    def momentum(self):
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum):
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__momentum = momentum

    @property
    def momentum_network(self):
        return self.__momentum_network

    @momentum_network.setter
    def momentum_network(self, momentum_network):
        self.__momentum_network = momentum_network

    @property
    def partial_momentum_network(self):
        return self.__partial_momentum_network

    @partial_momentum_network.setter
    def partial_momentum_network(self, partial_momentum_network):
        self.__partial_momentum_network = partial_momentum_network
    #endregion

    def apply_nesterov(self):
        # https://towardsdatascience.com/learning-parameters-part-2-a190bef2d12
        for grad_net_index in range(len(self.model.dense_configuration)):
            self.partial_momentum_network[grad_net_index]['weights'] *= self.momentum
            self.partial_momentum_network[grad_net_index]['biases'] *= self.momentum
            self.gradient_network[grad_net_index]['weights'] = np.subtract(
                self.gradient_network[grad_net_index]['weights'],
                self.partial_momentum_network[grad_net_index]['weights'])
            self.gradient_network[grad_net_index]['biases'] = np.subtract(
                self.gradient_network[grad_net_index]['biases'],
                self.partial_momentum_network[grad_net_index]['biases'])

    def apply_SGD(self):
        # update the weights only of the dense layers and store the in a struct to then update the true
        # vector
        for grad_net_index in range(len(self.model.dense_configuration)):
            layer_index = self.model.dense_configuration[grad_net_index]
            self.gradient_network[grad_net_index]['weights'] /= self.batch_size
            self.gradient_network[grad_net_index]['biases'] /= self.batch_size
            delta_w = self.lr * self.gradient_network[grad_net_index]['weights']
            delta_b = self.lr * self.gradient_network[grad_net_index]['biases']
            self.momentum_network[grad_net_index]['weights'] *= self.momentum
            self.momentum_network[grad_net_index]['biases'] *= self.momentum
            self.momentum_network[grad_net_index]['weights'] = np.add(self.momentum_network[grad_net_index]['weights'],
                                                                      delta_w)
            self.momentum_network[grad_net_index]['biases'] = np.add(self.momentum_network[grad_net_index]['biases'],
                                                                     delta_b)
            if self.model.layers[layer_index].regularizer is not None:
                # apply regularizers if the users specified some. These depende layer by layer so different
                # layers can have different regularizers
                self.model.layers[layer_index].weights = np.subtract(
                    np.add(self.model.layers[layer_index].weights, self.momentum_network[grad_net_index]['weights']),
                    self.model.layers[layer_index].regularizer.derive(
                        w=self.model.layers[layer_index].weights,
                        lambd=self.model.layers[layer_index].regularizer_param),
                )
            else:
                self.model.layers[layer_index].weights = np.add(
                    self.model.layers[layer_index].weights, self.momentum_network[grad_net_index]['weights'])

            self.model.layers[layer_index].biases = np.add(
                self.model.layers[layer_index].biases,
                self.momentum_network[grad_net_index]['biases']
            )
            self.partial_momentum_network = self.momentum_network

    def apply(self, *args):
        if self.nesterov:
            self.apply_nesterov()
        self.apply_SGD()

    def init_SGDnetwork_with_model(self, model):
        self.model = model
        self.gradient_network = self.model.get_empty_struct()
        self.momentum_network = self.model.get_empty_struct()
        self.partial_momentum_network = self.momentum_network

    def optimization_process(self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None):
        self.init_SGDnetwork_with_model(model)
        self.check_instances(early_stopping, check_stop)
        self.check_batch(train_dataset, train_labels, batch_size)
        self.do_epochs(validation, epochs, shuffle, early_stopping, optimizer=self.apply)
        return self.values_dict


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation
    https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
    """

    def __init__(self, learning_rate=0.01, rho=0.9):
        super().__init__(learning_rate, name='rmsprop')
        self.__rho = rho
        self.__epsilon = 1e-8
        self.rmsprop_network = ''

    #region RMSProp properties
    @property
    def rho(self):
        return self.__rho

    @property
    def rmsprop_network(self):
        return self.__rmsprop_network

    @rmsprop_network.setter
    def rmsprop_network(self, rmsprop_network):
        self.__rmsprop_network = rmsprop_network
    #endregion

    def init_rms_network_with_model(self, model):
        self.model = model
        self.rmsprop_network = self.model.get_empty_struct()
        self.gradient_network = self.model.get_empty_struct()

    def apply_rms(self, *args):
        for grad_net_index in range(len(self.model.dense_configuration)):
            layer_index = self.model.dense_configuration[grad_net_index]

            self.gradient_network[grad_net_index]['weights'] /= self.batch_size
            self.gradient_network[grad_net_index]['biases'] /= self.batch_size
            delta_w = self.gradient_network[grad_net_index]['weights']
            delta_b = self.gradient_network[grad_net_index]['biases']

            self.rmsprop_network[grad_net_index]['weights'] = np.add(
                self.rmsprop_network[grad_net_index]['weights'] * self.rho, np.power(delta_w, 2) * (1 - self.rho))
            self.rmsprop_network[grad_net_index]['biases'] = np.add(
                self.rmsprop_network[grad_net_index]['biases'] * self.rho, np.power(delta_b, 2) * (1 - self.rho))

            self.model.layers[layer_index].weights = np.add(self.model.layers[layer_index].weights,
                                                            self.lr * np.divide(delta_w, np.sqrt(
                                                                self.rmsprop_network[grad_net_index][
                                                                    'weights']) + self.__epsilon))
            self.model.layers[layer_index].biases = np.add(self.model.layers[layer_index].biases,
                                                           self.lr * np.divide(delta_b, np.sqrt(
                                                               self.rmsprop_network[grad_net_index][
                                                                   'biases']) + self.__epsilon))

    def optimization_process(self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None):
        self.init_rms_network_with_model(model)
        self.check_instances(early_stopping, check_stop)
        self.check_batch(train_dataset, train_labels, batch_size)
        self.do_epochs(validation, epochs, shuffle, early_stopping, optimizer=self.apply_rms)
        return self.values_dict


class Adam(Optimizer):
    """
    Adaptive Moment Estimation
    https://arxiv.org/abs/1412.6980
    """

    def __init__(self, learning_rate=0.1, momentum=0.00001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate, name='adam')
        self.__loss_function = ''
        self.__metric = ''
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon
        self.momentum = momentum
        self.momentum_network_1 = ''
        self.momentum_network_2 = ''

    #region Adam properties
    @property
    def momentum(self):
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum):
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__momentum = momentum

    @property
    def momentum_network_1(self):
        return self.__momentum_network_1

    @momentum_network_1.setter
    def momentum_network_1(self, momentum_network_1):
        self.__momentum_network_1 = momentum_network_1

    @property
    def momentum_network_2(self):
        return self.__momentum_network_2

    @momentum_network_2.setter
    def momentum_network_2(self, momentum_network_2):
        self.__momentum_network_2 = momentum_network_2
    #endregion

    def init_Adam_network_with_model(self, model):
        self.model = model
        self.gradient_network = self.model.get_empty_struct()
        self.momentum_network_1 = self.model.get_empty_struct()
        self.momentum_network_2 = self.model.get_empty_struct()

    def apply_adam(self, batch_index):
        for grad_net_index in range(len(self.model.dense_configuration)):
            layer_index = self.model.dense_configuration[grad_net_index]

            self.gradient_network[grad_net_index]['weights'] /= self.batch_size
            self.gradient_network[grad_net_index]['biases'] /= self.batch_size

            self.momentum_network_1[grad_net_index]['weights'] = np.add(
                self.__beta1 * self.momentum, self.gradient_network[grad_net_index]['weights'] * (1 - self.__beta1))
            self.momentum_network_2[grad_net_index]['biases'] = np.add(
                self.__beta1 * self.momentum, self.gradient_network[grad_net_index]['biases'] * (1 - self.__beta1))

            self.momentum_network_2[grad_net_index]['weights'] = np.add(
                self.__beta1 * self.momentum,
                np.power(self.gradient_network[grad_net_index]['weights'], 2) * (1 - self.__beta1))
            self.momentum_network_2[grad_net_index]['biases'] = np.add(
                self.__beta1 * self.momentum,
                np.power(self.gradient_network[grad_net_index]['biases'], 2) * (1 - self.__beta1))

            m_hat_w = np.divide(self.momentum_network_1[grad_net_index]['weights'],
                                (1 - np.power(self.__beta1, batch_index + 1)))
            v_hat_w = np.divide(self.momentum_network_2[grad_net_index]['weights'],
                                (1 - np.power(self.__beta2, batch_index + 1)))

            m_hat_b = np.divide(self.momentum_network_1[grad_net_index]['biases'],
                                (1 - np.power(self.__beta1, batch_index + 1)))
            v_hat_b = np.divide(self.momentum_network_2[grad_net_index]['biases'],
                                (1 - np.power(self.__beta2, batch_index + 1)))

            self.model.layers[layer_index].weights = np.add(self.model.layers[layer_index].weights,
                                                            self.lr * np.divide(m_hat_w,
                                                                                np.sqrt(v_hat_w + self.__epsilon)))
            self.model.layers[layer_index].biases = np.add(self.model.layers[layer_index].biases,
                                                           self.lr * np.divide(m_hat_b, np.sqrt(
                                                               v_hat_b + self.__epsilon)))

    def optimization_process(self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None):
        self.init_Adam_network_with_model(model)
        self.check_instances(early_stopping, check_stop)
        self.check_batch(train_dataset, train_labels, batch_size)
        self.init_Adam_network_with_model(model)
        self.do_epochs(validation, epochs, shuffle, early_stopping, optimizer=self.apply_adam)
        return self.values_dict


optimizers = {
    'sgd': StochasticGradientDescent,
    'rmsprop': RMSProp,
    'adam': Adam
}
