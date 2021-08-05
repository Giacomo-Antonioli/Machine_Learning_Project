import numpy as np

from Weight_Initializer import weights_initializers
from Reguralizers import regularizers

class Layer:
    """
    Class that represent a layer of a neural network
    Attributes:
        inp_dim: (int) layer's input dimension
        n_units: (int) number of units
        act: (str) name of the layer's activation function
        init_type: (str) weights initialization type ('fixed' or 'uniform')
        kwargs contains other attributes for the weights initialization
    """

    def __init__(self, input_dimension, n_units, weight_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',regularizer=None):  # ,activation_function, init_type, **kwargs):
        """ Constructor -> see parameters in the class description """
        self.weights = weights_initializers(init_type=weight_initializer, fan_in=input_dimension, fan_out=n_units)
        self.biases = weights_initializers(init_type=bias_initializer, fan_in=1, fan_out=n_units)
        self.__input_dimension = input_dimension
        self.__n_units = n_units
        # self.__activation_function = activation_functions[activation_function]
        self.__inputs = None
        self.__nets = None
        self.__outputs = None
        self.__gradient_w = None
        self.__gradient_b = None
        self.__type = 'layer'
        if regularizer==None:
            self.__regularizer=None
            self.__regularizer_param=None
        self.__regularizer=regularizer[0]
        self.__regularizer_param=regularizer[1]

    @property
    def get_dim(self):
        return self.__input_dimension

    @property
    def get_type(self):
        return self.__type

    @property
    def act(self):
        return self.__activation_function

    @property
    def n_units(self):
        return self.__n_units

    @property
    def inputs(self):
        return self.__inputs

    @property
    def nets(self):
        return self.__nets

    @property
    def outputs(self):
        return self.__outputs
    @property
    def regularizer(self):
        return self.__regularizer
    @property
    def regularizer_param(self):
        return self.__regularizer_param

    def forward_pass(self, input):
        """
        Performs the forward pass on the current layer
        :param input: (numpy ndarray) input vector
        :return: the vector of the current layer's outputs
        """
        self.__input = input
        self.__nets = np.add(np.matmul(self.__input, self.weights), self.biases)
        self.__outputs = self.__act.func(self.__nets)
        return self.__outputs

    def backward_pass(self, upstream_delta):
        """
        Sets the layer's gradients
        :param upstream_delta: for hidden layers, delta = dot_prod(delta_next, w_next) * dOut_dNet
            Multiply (dot product) already the delta for the current layer's weights in order to have it ready for the
            previous layer (that does not have access to this layer's weights) that will execute this method in the
            next iteration of Network.propagate_back()
        :return new_upstream_delta: delta already multiplied (dot product) by the current layer's weights
        :return gradient_w: gradient wrt weights
        :return gradient_b: gradient wrt biases
        """
        dOut_dNet = self.__activation_function.deriv(self.__nets)
        delta = np.multiply(upstream_delta, dOut_dNet)
        self.__gradient_b = -delta
        self.__gradient_w = np.zeros(shape=(self.__inp_dim, self.__n_units))
        for i in range(self.__input_dimension):
            for j in range(self.__n_units):
                self.__gradient_w[i][j] = -delta[j] * self.__inputs[i]
        # the i-th row of the weights matrix corresponds to the vector formed by the i-th weight of each layer's unit
        new_upstream_delta = [np.dot(delta, self.weights[i]) for i in range(self.__inp_dim)]
        return new_upstream_delta, self.__gradient_w, self.__gradient_b
