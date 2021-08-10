import numpy as np

from NeuralNetworkCore.Activations import activation_functions
from NeuralNetworkCore.Weight_Initializer import weights_initializers


class Layer:
    """
    Class that represent a layer of a neural network
    """

    def __init__(self, type):
        self.__type = type

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, type):
        self.__type = type


class Dense(Layer):
    """
    Class that represent a dense layer of a neural network
    """

    def __init__(self, input_dimension, n_units, type='dense',weight_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                 regularizer=None, activation_function='relu'):
        """
        Function constructor of a dense layer
        :param input_dimension:
        :param n_units:
        :param weight_initializer:
        :param bias_initializer:
        :param regularizer:
        :param activation_function:
        """
        super().__init__(type)
        self.weights = weights_initializers(init_type=weight_initializer, fan_in=input_dimension, fan_out=n_units)
        self.biases = weights_initializers(init_type=bias_initializer, fan_in=1, fan_out=n_units)
        self.__input_dimension = input_dimension
        self.__n_units = n_units
        self.__activation_function = activation_functions[activation_function]
        self.__inputs = None
        self.__nets = None
        self.__outputs = None
        self.__gradient_w = None
        self.__gradient_b = None

        if regularizer == None:
            self.__regularizer = None
            self.__regularizer_param = None
        else:
            self.__regularizer = regularizer[0]
            self.__regularizer_param = regularizer[1]

    @property
    def get_dim(self):
        return self.__input_dimension



    @property
    def activation_function(self):
        return self.__activation_function

    @activation_function.setter
    def activation_function(self, activation_function):
        self.__activation_function = activation_function

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
        self.__inputs = input
        self.__nets = np.add(np.matmul(self.__inputs, self.weights), self.biases)
        self.__outputs = self.activation_function.function(self.__nets)
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
        dOut_dNet = self.__activation_function.derive(self.__nets)
        delta = np.multiply(upstream_delta, dOut_dNet)
        self.__gradient_b = -delta
        self.__gradient_w = np.zeros(shape=(self.__input_dimension, self.__n_units))
        for i in range(self.__input_dimension):
            for j in range(self.__n_units):
                self.__gradient_w[i][j] = -delta[j] * self.__inputs[i]
        # the i-th row of the weights matrix corresponds to the vector formed by the i-th weight of each layer's unit
        new_upstream_delta = [np.dot(delta, self.weights[i]) for i in range(self.__input_dimension)]
        return new_upstream_delta, self.__gradient_w, self.__gradient_b


class Dropout(Layer):
    """
    Class representing a dropout layer of a neural network
    """

    def __init__(self, probability=0, seed=42):
        """
        Constructor of the Dropout layer.
        :param probability: probability percentage of an input to not be passed on
        :param seed: seed of the randomization
        """
        super().__init__('drop')
        self.__original_inputs = None
        self.__outputs=[0]
        self.__probability = probability
        self.__seed = seed
        self.__rng_generator = np.random.default_rng(seed)

    @property
    def probability(self):
        return self.__probability

    @probability.setter
    def probability(self, probability):
        self.__probability = probability

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = seed

    @property
    def rng_generator(self):
        return self.__rng_generator

    @rng_generator.setter
    def rng_generator(self, rng_generator):
        self.__rng_generator = rng_generator

    @property
    def original_inputs(self):
        return self.__original_inputs

    @original_inputs.setter
    def original_inputs(self, original_inputs):
        self.__original_inputs = original_inputs

    @property
    def outputs(self):
        return self.__outputs

    @outputs.setter
    def outputs(self, outputs):
        self.__outputs = outputs

    def forward_pass(self, input):
        """
        Performs the forward pass on the current layer
        :param input: (numpy ndarray) input vector
        :return: the vector of the current layer's outputs
        """
        self.original_inputs = input
        self.outputs=self.original_inputs
        for x in range(len(self.original_inputs)):
            if self.rng_generator.uniform()<self.probability:
                self.outputs[x]=0
        return self.outputs

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

        return upstream_delta,0,0