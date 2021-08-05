import math
from abc import ABC, abstractmethod

import numpy as np
import tqdm as tqdm
from Loss import losses
from Metrics import metrics
from LearningRates import lr_decays
from Reguralizers import regularizers


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    (check 'ABC' documentation for more info about abstract classes in Python)

    Attributes:
        model: ('Network' object) Neural Network to which apply the algorithm
        loss: ('DerivableFunction' object) loss function
        metric: ('Function' object) accuracy function
        lr: (float) learning rate
        base_lr: (float) the initial learning rate
        final_lr: (float) the final learning rate in case of linear decay (it's 1% of base_le)
        lr_decay: (str) the type of learning rate decay
        limit_step: (int) the number of the weight update where the linear decaying learning rate has to reach final_lr
        decay_rate: (float) for exponential learning rate decay: the higher, the stronger the decay
        decay_steps: (int) similar to limit_step but for the exponential decay case
        staircase: (bool) if True, the exponentially decaying learning rate decays in a stair-like fashion
        momentum: (float) momentum coefficient
        lambd: (float) regularization coefficient
        reg_type: (str) the type of regularization (either None, 'l1', 'l2')
    """

    @abstractmethod
    def __init__(self):
        self.__type = 'optimizer'






class StochasticGradientDescent(Optimizer):
    """
    Stochastic Gradient Descent
    Concrete implementation of the abstract class Optimizer
    """

    def __init__(self, learning_rate=0.01, momentum=0, nesterov=False, loss_function='squared', metrics='euclidean'):
        super().__init__()
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError('learning_rate should be a value between 0 and 1, Got:{}'.format(learning_rate))
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__lr = learning_rate
        self.__momentum = momentum
        self.__nesterov = nesterov
        self.__loss_function = loss_function
        self.__metrics = metrics
        self.__name = 'sgd'

    @property
    def type(self):
        return self.__type

    @property
    def name(self):
        return self.__name
    @property
    def loss_function(self):
        return self.__loss_function

    @property
    def metrics(self):
        return self.__metrics

    @property
    def nesterov(self):
        return self.__nesterov

    def optimization_process(self, model,train_dataset, train_labels, epochs=1, batch_size=1, shuffle=False, validation=None):

        train_dataset = train_dataset[np.newaxis, :] if len(train_dataset.shape) < 2 else train_dataset
        train_labels = train_labels[np.newaxis, :] if len(train_labels.shape) < 2 else train_labels

        values_dict = {'training_error': [],
                       'training_metrics': [],
                       'validation_error': [],
                       'validation_metrics': []}

        momentum_network = model.get_empty_struct()
        step = 0

        # cycle through epochs
        for _ in tqdm.tqdm(range(epochs), desc="Training"):
            epoch_training_error = np.zeros(model.layers[-1].n_units)
            epoch_training_error_metric = np.zeros(model.layers[-1].n_units)

            if batch_size != train_dataset.shape[0] and shuffle:
                indexes = list(range(len(train_dataset)))
                np.random.shuffle(indexes)
                train_dataset = train_dataset[indexes]
                train_labels = train_labels[indexes]

            # cycle through batches
            for batch_index in tqdm.tqdm(range(math.ceil(len(train_dataset) / batch_size)),
                                         desc="Cycling though " + math.ceil(
                                             len(train_dataset) / batch_size) + "batches"):
                start = batch_index * batch_size
                end = start + batch_size
                train_batch = train_dataset[start: end]
                targets_batch = train_labels[start: end]
                gradient_network = model.get_empty_struct()

                for current_input, current_target in zip(train_batch, targets_batch):
                    net_outputs = model.forward(inp=current_input)

                    # epoch training error = itself + loss + regularization
                    epoch_training_error = np.add(epoch_training_error,
                                                  self.__loss_function.func(predicted=net_outputs, target=current_target))

                    epoch_training_error_metric = np.add(epoch_training_error_metric,
                                                         self.__metrics.func(predicted=net_outputs, target=current_target))


                    dErr_dOut = self.__loss_function.deriv(predicted=net_outputs, target=current_target)
                    gradient_network = model.propagate_back(dErr_dOut, gradient_network)

                # # learning rate decays
                # if self.lr_decay is not None:
                #     step += 1
                #     self.lr = lr_decays[self.lr_decay].func(curr_step=step, **self.lr_params)

                # weights update
                for layer_index in range(len(model.layers)):
                    # gradient_network contains the gradients of all the layers (and units) in the network
                    gradient_network[layer_index]['weights'] /= batch_size
                    gradient_network[layer_index]['biases'] /= batch_size
                    # delta_w is equivalent to lrn_rate * local_grad * input_on_that_connection (local_grad = delta)
                    delta_w = self.lr * gradient_network[layer_index]['weights']
                    delta_b = self.lr * gradient_network[layer_index]['biases']
                    # momentum_network[layer_index]['weights'] is the new delta_w --> it adds the momentum
                    # Since it acts as delta_w, it multiplies itself by the momentum constant and then adds
                    # lrn_rate * local_grad * input_on_that_connection (i.e. "delta_w")
                    momentum_network[layer_index]['weights'] *= self.momentum
                    momentum_network[layer_index]['biases'] *= self.momentum
                    momentum_network[layer_index]['weights'] = np.add(momentum_network[layer_index]['weights'], delta_w)
                    momentum_network[layer_index]['biases'] = np.add(momentum_network[layer_index]['biases'], delta_b)
                    if model.layers[layer_index].regularizer != None:
                        model.layers[layer_index].weights = np.subtract(
                            np.add(model.layers[layer_index].weights, momentum_network[layer_index]['weights']),
                            regularizers[model.layers[layer_index].regularizer].deriv(w=model.layers[layer_index].weights, lambd=smodel.layers[layer_index].regularizer_param),
                        )
                    model.layers[layer_index].biases = np.add(
                        model.layers[layer_index].biases,
                        momentum_network[layer_index]['biases']
                    )

            # validation
            if validation is not None:
                val_x = validation[0][np.newaxis, :] if len(validation[0].shape) < 2 else validation[0]
                val_y = validation[1][np.newaxis, :] if len(validation[1].shape) < 2 else validation[1]

                epoch_val_error, epoch_val_metric = model.evaluate(inp=val_x, targets=val_y, metr=self.__metrics.name,
                                                                 loss=self.__loss_function.name)
                values_dict['validation_error'].append(epoch_val_error)
                values_dict['validation_metrics'].append(epoch_val_metric)

            epoch_training_error = np.sum(epoch_training_error) / len(epoch_training_error)
            values_dict['training_error'].append(epoch_training_error / len(train_dataset))
            epoch_training_error_metric = np.sum(epoch_training_error_metric) / len(epoch_training_error_metric)
            values_dict['training_metrics'].append(epoch_training_error_metric / len(train_dataset))

        return values_dict
        #################################


optimizers = {
    'sgd': StochasticGradientDescent
}
