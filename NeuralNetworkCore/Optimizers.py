from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import math
from NeuralNetworkCore.Loss import losses
from NeuralNetworkCore.Metrics import metrics


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    """

    @abstractmethod
    def __init__(self):
        self.__type = 'optimizer'


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
        super().__init__()
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError('learning_rate should be a value between 0 and 1, Got:{}'.format(learning_rate))
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__lr = learning_rate
        self.__momentum = momentum
        self.__nesterov = nesterov
        self.__name = 'sgd'
        self.__loss_function = ''
        self.__metric = ''

    @property
    def type(self):
        return self.__type

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = lr

    @property
    def momentum(self):
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum):
        self.__momentum = momentum

    @property
    def name(self):
        return self.__name

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
    def nesterov(self):
        return self.__nesterov

    def optimization_process(self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None):
        if early_stopping == True and check_stop is None:
            raise AttributeError("CheckStop value is invalid")
        stop_variable = False

        if isinstance(model.loss, str):
            self.loss_function = losses[model.loss]
        else:
            self.loss_function = model.loss

        if isinstance(model.metrics, str):
            self.metric = metrics[model.metrics]
        else:
            self.metric = model.metrics
        if batch_size == None:
            batch_size = 1
        if batch_size== 'all':
            batch_size=len(train_dataset)
        train_dataset = train_dataset[np.newaxis, :] if len(train_dataset.shape) < 2 else train_dataset
        train_labels = train_labels[np.newaxis, :] if len(train_labels.shape) < 2 else train_labels

        values_dict = {'training_error': [],
                       'training_metrics': [],
                       'validation_error': [],
                       'validation_metrics': []}

        momentum_network = model.get_empty_struct()
        partial_momentum_network = momentum_network
        current_val_error = 0
        epoch = 0
        pbar = tqdm(total=epochs)
        while epoch < epochs and not stop_variable:

            epoch_training_error = np.zeros(model.layers[-1].n_units)
            epoch_training_error_metric = np.zeros(model.layers[-1].n_units)

            if batch_size != train_dataset.shape[0] and shuffle:
                indexes = list(range(len(train_dataset)))
                np.random.shuffle(indexes)
                train_dataset = train_dataset[indexes]
                train_labels = train_labels[indexes]

            for batch_index in range(math.ceil(len(train_dataset) / batch_size)):
                start = batch_index * batch_size
                end = start + batch_size
                train_batch = train_dataset[start: end]
                targets_batch = train_labels[start: end]
                gradient_network = model.get_empty_struct()
                for current_input, current_target in zip(train_batch, targets_batch):
                    net_outputs = model.forward(net_input=current_input, training=True)


                    epoch_training_error = np.add(epoch_training_error,
                                                  self.__loss_function.function(predicted=net_outputs,
                                                                                target=current_target))

                    epoch_training_error_metric = np.add(epoch_training_error_metric,
                                                         self.__metric.function(predicted=net_outputs,
                                                                                target=current_target))

                    dErr_dOut = self.loss_function.derive(predicted=net_outputs, target=current_target)

                    gradient_network = model.propagate_back(dErr_dOut, gradient_network)
                if self.nesterov:
                    # https://towardsdatascience.com/learning-parameters-part-2-a190bef2d12
                    for grad_net_index in range(len(model.dense_configuration)):
                        partial_momentum_network[grad_net_index]['weights'] *= self.momentum
                        partial_momentum_network[grad_net_index]['biases'] *= self.momentum
                        gradient_network[grad_net_index]['weights'] = np.subtract(
                            gradient_network[grad_net_index]['weights'],
                            partial_momentum_network[grad_net_index]['weights'])
                        gradient_network[grad_net_index]['biases'] = np.subtract(
                            gradient_network[grad_net_index]['biases'],
                            partial_momentum_network[grad_net_index]['biases'])
                for grad_net_index in range(len(model.dense_configuration)):

                    layer_index = model.dense_configuration[grad_net_index]

                    gradient_network[grad_net_index]['weights'] /= batch_size
                    gradient_network[grad_net_index]['biases'] /= batch_size
                    delta_w = self.lr * gradient_network[grad_net_index]['weights']
                    delta_b = self.lr * gradient_network[grad_net_index]['biases']
                    momentum_network[grad_net_index]['weights'] *= self.momentum
                    momentum_network[grad_net_index]['biases'] *= self.momentum
                    momentum_network[grad_net_index]['weights'] = np.add(momentum_network[grad_net_index]['weights'],
                                                                         delta_w)
                    momentum_network[grad_net_index]['biases'] = np.add(momentum_network[grad_net_index]['biases'],
                                                                        delta_b)

                    if model.layers[layer_index].regularizer != None:

                        model.layers[layer_index].weights = np.subtract(
                            np.add(model.layers[layer_index].weights, momentum_network[grad_net_index]['weights']),
                            model.layers[layer_index].regularizer.derive(
                                w=model.layers[layer_index].weights,
                                lambd=model.layers[layer_index].regularizer_param),
                        )
                    else:
                        model.layers[layer_index].weights = np.add(
                            model.layers[layer_index].weights, momentum_network[grad_net_index]['weights'])

                    model.layers[layer_index].biases = np.add(
                        model.layers[layer_index].biases,
                        momentum_network[grad_net_index]['biases']
                    )
                    partial_momentum_network = momentum_network
            if validation is not None:
                val_x = validation[0][np.newaxis, :] if len(validation[0].shape) < 2 else validation[0]
                val_y = validation[1][np.newaxis, :] if len(validation[1].shape) < 2 else validation[1]
                epoch_val_error, epoch_val_metric = model.evaluate(validation_data=val_x, targets=val_y)
                current_val_error = epoch_val_error
                values_dict['validation_error'].append(current_val_error)
                values_dict['validation_metrics'].append(epoch_val_metric)

            epoch_training_error = np.sum(epoch_training_error) / len(epoch_training_error)
            current_loss_error = epoch_training_error / len(train_dataset)
            values_dict['training_error'].append(current_loss_error)
            epoch_training_error_metric = np.sum(epoch_training_error_metric) / len(epoch_training_error_metric)
            values_dict['training_metrics'].append(epoch_training_error_metric / len(train_dataset))
            if early_stopping:
                if check_stop.monitor == 'loss':
                    stop_variable = check_stop.apply_stop(current_loss_error)
                elif check_stop.monitor == 'val':
                    stop_variable = check_stop.apply_stop(current_val_error)
            epoch += 1
            pbar.update(1)
        pbar.close()
        return values_dict


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation
    """

    def __init__(self, learning_rate=0.01, momentum=0.1, rho=0.9):
        super().__init__()
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError('learning_rate should be a value between 0 and 1, Got:{}'.format(learning_rate))
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__lr = learning_rate
        self.__momentum = momentum
        self.__rho = rho
        self.__name = 'rmsprop'
        self.__loss_function = ''
        self.__metric = ''
        self.__epsilon=1e-8

    @property
    def type(self):
        return self.__type

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = lr

    @property
    def momentum(self):
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum):
        self.__momentum = momentum

    @property
    def name(self):
        return self.__name

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
    def rho(self):
        return self.__rho

    def optimization_process(self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None):
        if early_stopping == True and check_stop is None:
            raise AttributeError("CheckStop value is invalid")
        stop_variable = False
        print(model.loss)
        if isinstance(model.loss, str):
            self.loss_function = losses[model.loss]
        else:
            self.loss_function = model.loss

        if isinstance(model.metrics, str):
            self.metric = metrics[model.metrics]
        else:
            self.metric = model.metrics
        if batch_size == None:
            batch_size = 1
        if batch_size== 'all':
            batch_size = 1
        train_dataset = train_dataset[np.newaxis, :] if len(train_dataset.shape) < 2 else train_dataset
        train_labels = train_labels[np.newaxis, :] if len(train_labels.shape) < 2 else train_labels

        values_dict = {'training_error': [],
                       'training_metrics': [],
                       'validation_error': [],
                       'validation_metrics': []}

        rmsprop_network = model.get_empty_struct()

        current_val_error = 0
        epoch = 0
        pbar = tqdm(total=epochs)
        while epoch < epochs and not stop_variable:

            epoch_training_error = np.zeros(model.layers[-1].n_units)
            epoch_training_error_metric = np.zeros(model.layers[-1].n_units)

            if batch_size != train_dataset.shape[0] and shuffle:
                indexes = list(range(len(train_dataset)))
                np.random.shuffle(indexes)
                train_dataset = train_dataset[indexes]
                train_labels = train_labels[indexes]

            for batch_index in range(math.ceil(len(train_dataset) / batch_size)):
                start = batch_index * batch_size
                end = start + batch_size
                train_batch = train_dataset[start: end]
                targets_batch = train_labels[start: end]
                gradient_network = model.get_empty_struct()
                for current_input, current_target in zip(train_batch, targets_batch):
                    net_outputs = model.forward(net_input=current_input, training=True)


                    epoch_training_error = np.add(epoch_training_error,
                                                  self.__loss_function.function(predicted=net_outputs,
                                                                                target=current_target))

                    epoch_training_error_metric = np.add(epoch_training_error_metric,
                                                         self.__metric.function(predicted=net_outputs,
                                                                                target=current_target))

                    dErr_dOut = self.__loss_function.derive(predicted=net_outputs, target=current_target)
                    gradient_network = model.propagate_back(dErr_dOut, gradient_network)

                for grad_net_index in range(len(model.dense_configuration)):
                    layer_index = model.dense_configuration[grad_net_index]

                    gradient_network[grad_net_index]['weights'] /= batch_size
                    gradient_network[grad_net_index]['biases'] /= batch_size
                    delta_w = gradient_network[grad_net_index]['weights']
                    delta_b = gradient_network[grad_net_index]['biases']

                    rmsprop_network[grad_net_index]['weights'] = np.add(
                        rmsprop_network[grad_net_index]['weights'] * self.rho, np.power(delta_w, 2) * (1 - self.rho))
                    rmsprop_network[grad_net_index]['biases'] = np.add(
                        rmsprop_network[grad_net_index]['biases'] * self.rho, np.power(delta_b, 2) * (1 - self.rho))

                    model.layers[layer_index].weights = np.add(model.layers[layer_index].weights,
                                                               self.lr * np.divide(delta_w, np.sqrt(
                                                                   rmsprop_network[grad_net_index]['weights'])+self.__epsilon))
                    model.layers[layer_index].biases = np.add(model.layers[layer_index].biases,
                                                              self.lr * np.divide(delta_b, np.sqrt(
                                                                  rmsprop_network[grad_net_index]['biases'])+self.__epsilon))


            if validation is not None:
                val_x = validation[0][np.newaxis, :] if len(validation[0].shape) < 2 else validation[0]
                val_y = validation[1][np.newaxis, :] if len(validation[1].shape) < 2 else validation[1]
                epoch_val_error, epoch_val_metric = model.evaluate(validation_data=val_x, targets=val_y)
                current_val_error = epoch_val_error
                values_dict['validation_error'].append(current_val_error)
                values_dict['validation_metrics'].append(epoch_val_metric)

            epoch_training_error = np.sum(epoch_training_error) / len(epoch_training_error)
            current_loss_error = epoch_training_error / len(train_dataset)
            values_dict['training_error'].append(current_loss_error)
            epoch_training_error_metric = np.sum(epoch_training_error_metric) / len(epoch_training_error_metric)
            values_dict['training_metrics'].append(epoch_training_error_metric / len(train_dataset))
            if early_stopping:
                if check_stop.monitor == 'loss':
                    stop_variable = check_stop.apply_stop(current_loss_error)
                elif check_stop.monitor == 'val':
                    stop_variable = check_stop.apply_stop(current_val_error)
            epoch += 1
            pbar.update(1)
        pbar.close()
        return values_dict



class Adam(Optimizer):
    """
    Adaptive Moment Estimation
    """

    def __init__(self, learning_rate=0.1, momentum=0.00001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError('learning_rate should be a value between 0 and 1, Got:{}'.format(learning_rate))
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__lr = learning_rate
        self.__momentum = momentum
        self.__name = 'adam'
        self.__loss_function = ''
        self.__metric = ''
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon

    @property
    def type(self):
        return self.__type

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = lr

    @property
    def momentum(self):
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum):
        self.__momentum = momentum

    @property
    def name(self):
        return self.__name

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
    def rho(self):
        return self.__rho

    def optimization_process(self, model, train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False,
                             validation=None, early_stopping=False, check_stop=None):
        if early_stopping == True and check_stop is None:
            raise AttributeError("CheckStop value is invalid")
        stop_variable = False
        print(model.loss)
        if isinstance(model.loss, str):
            self.loss_function = losses[model.loss]
        else:
            self.loss_function = model.loss

        if isinstance(model.metrics, str):
            self.metric = metrics[model.metrics]
        else:
            self.metric = model.metrics
        if batch_size == None:
            batch_size = 1
        if batch_size== 'all':
            batch_size=len(train_dataset)

        train_dataset = train_dataset[np.newaxis, :] if len(train_dataset.shape) < 2 else train_dataset
        train_labels = train_labels[np.newaxis, :] if len(train_labels.shape) < 2 else train_labels

        values_dict = {'training_error': [],
                       'training_metrics': [],
                       'validation_error': [],
                       'validation_metrics': []}

        momentum_network_1 = model.get_empty_struct()
        momentum_network_2 = model.get_empty_struct()
        current_val_error = 0
        epoch = 0
        pbar = tqdm(total=epochs)
        while epoch < epochs + 1 and not stop_variable:

            epoch_training_error = np.zeros(model.layers[-1].n_units)
            epoch_training_error_metric = np.zeros(model.layers[-1].n_units)

            if batch_size != train_dataset.shape[0] and shuffle:
                indexes = list(range(len(train_dataset)))
                np.random.shuffle(indexes)
                train_dataset = train_dataset[indexes]
                train_labels = train_labels[indexes]

            for batch_index in range(math.ceil(len(train_dataset) / batch_size)):
                start = batch_index * batch_size
                end = start + batch_size
                train_batch = train_dataset[start: end]
                targets_batch = train_labels[start: end]
                gradient_network = model.get_empty_struct()
                for current_input, current_target in zip(train_batch, targets_batch):
                    net_outputs = model.forward(net_input=current_input, training=True)
                    epoch_training_error = np.add(epoch_training_error,
                                                  self.__loss_function.function(predicted=net_outputs,
                                                                                target=current_target))

                    epoch_training_error_metric = np.add(epoch_training_error_metric,
                                                         self.__metric.function(predicted=net_outputs,
                                                                                target=current_target))

                    dErr_dOut = self.__loss_function.derive(predicted=net_outputs, target=current_target)
                    gradient_network = model.propagate_back(dErr_dOut, gradient_network)


                for grad_net_index in range(len(model.dense_configuration)):
                    layer_index = model.dense_configuration[grad_net_index]

                    gradient_network[grad_net_index]['weights'] /= batch_size
                    gradient_network[grad_net_index]['biases'] /= batch_size

                    momentum_network_1[grad_net_index]['weights'] = np.add(
                        self.__beta1 * self.momentum, gradient_network[grad_net_index]['weights'] * (1 - self.__beta1))
                    momentum_network_2[grad_net_index]['biases'] = np.add(
                        self.__beta1 * self.momentum, gradient_network[grad_net_index]['biases'] * (1 - self.__beta1))

                    momentum_network_2[grad_net_index]['weights'] = np.add(
                        self.__beta1 * self.momentum,
                        np.power(gradient_network[grad_net_index]['weights'], 2) * (1 - self.__beta1))
                    momentum_network_2[grad_net_index]['biases'] = np.add(
                        self.__beta1 * self.momentum,
                        np.power(gradient_network[grad_net_index]['biases'], 2) * (1 - self.__beta1))

                    m_hat_w = np.divide(momentum_network_1[grad_net_index]['weights'],
                                        (1 - np.power(self.__beta1, batch_index + 1)))
                    v_hat_w = np.divide(momentum_network_2[grad_net_index]['weights'],
                                        (1 - np.power(self.__beta2, batch_index + 1)))

                    m_hat_b = np.divide(momentum_network_1[grad_net_index]['biases'],
                                        (1 - np.power(self.__beta1, batch_index + 1)))
                    v_hat_b = np.divide(momentum_network_2[grad_net_index]['biases'],
                                        (1 - np.power(self.__beta2, batch_index + 1)))

                    model.layers[layer_index].weights = np.add(model.layers[layer_index].weights,
                                                               self.lr * np.divide(m_hat_w,
                                                                                   np.sqrt(v_hat_w + self.__epsilon)))
                    model.layers[layer_index].biases = np.add(model.layers[layer_index].biases,
                                                              self.lr * np.divide(m_hat_b, np.sqrt(
                                                                  v_hat_b + self.__epsilon)))

            if validation is not None:
                val_x = validation[0][np.newaxis, :] if len(validation[0].shape) < 2 else validation[0]
                val_y = validation[1][np.newaxis, :] if len(validation[1].shape) < 2 else validation[1]
                epoch_val_error, epoch_val_metric = model.evaluate(validation_data=val_x, targets=val_y)
                current_val_error = epoch_val_error
                values_dict['validation_error'].append(current_val_error)
                values_dict['validation_metrics'].append(epoch_val_metric)

            epoch_training_error = np.sum(epoch_training_error) / len(epoch_training_error)
            current_loss_error = epoch_training_error / len(train_dataset)
            values_dict['training_error'].append(current_loss_error)
            epoch_training_error_metric = np.sum(epoch_training_error_metric) / len(epoch_training_error_metric)
            values_dict['training_metrics'].append(epoch_training_error_metric / len(train_dataset))
            if early_stopping:
                if check_stop.monitor == 'loss':
                    stop_variable = check_stop.apply_stop(current_loss_error)
                elif check_stop.monitor == 'val':
                    stop_variable = check_stop.apply_stop(current_val_error)
            epoch += 1
            pbar.update(1)
        pbar.close()
        return values_dict


optimizers = {
    'sgd': StochasticGradientDescent,
    'rmsprop': RMSProp,
    'adam': Adam
}
