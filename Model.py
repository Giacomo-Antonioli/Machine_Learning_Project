import numpy as np
from Optimizers import *

class Model():
    def __init__(self, name):
        self.__name = name
        self.__layers = []
        self.__optimizer = None

    @property
    def layers(self):
        return self.__layers

    @property
    def name(self):
        return self.__name

    @property
    def optimizer(self):
        return self.__optimizer

    def add(self, object):
        if object.type == 'layer':
            self.__layers.append(object)


    def get_empty_struct(self):
        """ :return: a zeroed structure with the same topology of the NN to contain all the layers' gradients """
        struct = np.array([{}] * len(self.__layers))
        for layer_index in range(len(self.__layers)):
            struct[layer_index] = {'weights': [], 'biases': []}
            weights_matrix = self.__layers[layer_index].weights
            weights_matrix = weights_matrix[np.newaxis, :] if len(weights_matrix.shape) < 2 else weights_matrix
            struct[layer_index]['weights'] = np.zeros(shape=weights_matrix.shape)
            struct[layer_index]['biases'] = np.zeros(shape=(len(weights_matrix[0, :])))
        return struct

    def forward(self, net_input):
        """
        Performs a forward pass on the whole Network
        :param inp: net's input vector/matrix
        :return: net's output vector/matrix
        """
        output = net_input
        for layer in self.__layers:
            net_input = output
            output = layer.forward_pass(net_input)
        return output

    def compile(self, optimizer='sgd', loss='squared', metrics='bin_class_acc'):
        """
        Prepares the network for training by assigning an optimizer to it and setting its parameters
        :param opt: ('Optimizer' object)
        :param loss: (str) the type of loss function
        :param metr: (str) the type of metric to track (accuracy etc)
        :param lr: (float) learning rate value
        :param lr_decay: type of decay for the learning rate
        :param decay_rate: (float) the higher, the stronger the exponential decay
        :param decay_steps: (int) number of steps taken to decay the learning rate exponentially
        :param staircase: (bool) if True the learning rate will decrease in a stair-like fashion (used w/ exponential)
        :param limit_step: number of steps of weights update to perform before stopping decaying the learning rate
        :param momentum: (float) momentum parameter
        :param lambd: (float) regularization parameter
        :param reg_type: (string) regularization type
        """
        if isinstance(optimizer,str):
            self.__optimizer=optimizers[optimizer]
        elif isinstance(optimizer,Optimizer):
            self.__optimizer=optimizer

        self.__loss=loss
        self.__metrics=metrics



        # self.__params = {**self.__params, **{'loss': loss, 'metr': metr, 'lr': lr, 'lr_decay': lr_decay,
        #                                      'limit_step': limit_step, 'decay_rate': decay_rate,
        #                                      'decay_steps': decay_steps, 'staircase': staircase, 'momentum': momentum,
        #                                      'reg_type': reg_type, 'lambd': lambd}}
        # self.__opt = optimizers[opt](net=self, loss=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
        #                              decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase,
        #                              momentum=momentum, reg_type=reg_type, lambd=lambd)
