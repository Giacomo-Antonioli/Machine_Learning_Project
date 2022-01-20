from NeuralNetworkCore.Layer import Dense, Dropout
from NeuralNetworkCore.Optimizers import *
from NeuralNetworkCore.Reguralizers import EarlyStopping, regularizers


class Model:

    @staticmethod
    def compile_default():
        return 'sgd', 'squared', 'mee'

    def __init__(self, name="NN"):
        self.__name = name
        self.__layers = []
        self.__optimizer = None
        self.__loss = None
        self.__metrics = None
        self.__training_data = None
        self.__training_targets = None
        self.__validation_data = None
        self.__epochs = 1
        self.__batch_size = None
        self.__validation_split = 0
        self.__dense_configuration = []
        self.__net_configuration_types = []
        self.__early_stopping = False
        self.__check_stop = None
        self.__input_shape = 0

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, layers):
        self.__layers = layers

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss):
        self.__loss = loss

    @property
    def metrics(self):
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics):
        self.__metrics = metrics

    @property
    def training_data(self):
        return self.__training_data

    @training_data.setter
    def training_data(self, training_data):
        self.__training_data = training_data

    @property
    def training_targets(self):
        return self.__training_targets

    @training_targets.setter
    def training_targets(self, training_targets):
        self.__training_targets = training_targets

    @property
    def validation_data(self):
        return self.__validation_data

    @validation_data.setter
    def validation_data(self, validation_data):
        self.__validation_data = validation_data

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs):
        self.__epochs = epochs

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.__batch_size = batch_size

    @property
    def validation_split(self):
        return self.__validation_split

    @validation_split.setter
    def validation_split(self, validation_split):
        self.__validation_split = validation_split

    @property
    def dense_configuration(self):
        return self.__dense_configuration

    @dense_configuration.setter
    def dense_configuration(self, dense_configuration):
        self.__dense_configuration = dense_configuration

    @property
    def net_configuration_types(self):
        return self.__net_configuration_types

    @net_configuration_types.setter
    def net_configuration_types(self, net_configuration_types):
        self.__net_configuration_types = net_configuration_types

    @property
    def input_shape(self):
        return self.__input_shape

    def set_input_shape(self, input_shape):
        self.__input_shape = input_shape

    def add(self, object):
        self.layers.append(object)
        if object.type == 'dense':
            self.dense_configuration.append(self.get_layer_depth())
        self.net_configuration_types.append(object.type)

    def get_layer_depth(self):
        return len(self.layers) - 1
    
    def create_net(self, num_layer=1, drop_frequency=None, num_unit=[4], act_func=['linear'], weight_init=['glorot_uniform'], regularizer=None,
                   bias_init=['glorot_uniform'],  drop_percentage=[0.3], drop_seed=[10]):
        drop_count = 0
        for i in range(num_layer):
            #adding a dropout layer
            if drop_frequency != None and drop_count == drop_frequency and i != num_layer-1:
                seed = drop_seed.pop(0)
                prob = drop_percentage.pop(0)
                self.add(Dropout(prob))
                drop_seed.append(seed)
                drop_percentage.append(prob)
                drop_count = 0
            #adding a dense layer
            else:
                drop_count+=1
                n_unit = num_unit.pop(0) if i != num_layer-1 else 1
                activation_function = act_func.pop(0)
                w_init = weight_init.pop(0)
                b_init = bias_init.pop(0)
                reg = None
                if regularizer != None: 
                    reg = regularizer.pop(0)
                    regularizer.append(reg)
                self.add(Dense(n_units=n_unit, weight_initializer=w_init, bias_initializer=b_init, activation_function=activation_function, regularizer=reg))
                if i != num_layer-1: 
                    num_unit.append(n_unit)
                    act_func.append(activation_function)
                    weight_init.append(w_init)
                    bias_init.append(b_init)

    def showLayers(self):
        print("°°°°°°°°°°°°°°°°°°°°°")
        print('° Model Configuration')
        for x in self.__layers:
            if x.type == 'dense':
                print('° Dense Layer: ' + str(x.input_dimension) + ' inputs °')
                print('° Dense Layer: ' + str(x.n_units) + ' units °')
                print('° Dense Layer: ' + str(x.weights.shape) + ' internal dims °')
                if x.regularizer != None:
                    print('° Regularizer: ' + list(regularizers.keys())[
                        list(regularizers.values()).index(x.regularizer)] + ' °')
                    print('° Regularizer param: ' + str(x.regularizer_param) + ' °')
            elif x.type == 'drop':
                print('° Dropout Layer: ' + str(x.probability * 100) + '% °')
        print("°°°°°°°°°°°°°°°°°°°°°")

    def get_empty_struct(self):
        """ :return: a zeroed structure with the same topology of the NN to contain all the layers' gradients """
        valid_layers = 0
        for layer in self.layers:
            if layer.type == 'dense':
                valid_layers += 1
        struct = np.array([{}] * valid_layers)
        layer_index = 0
        for index, layer in enumerate(self.layers):
            if layer.type == 'dense':
                struct[layer_index] = {'weights': [], 'biases': []}
                weights_matrix = self.layers[index].weights
                weights_matrix = weights_matrix[np.newaxis, :] if len(weights_matrix.shape) < 2 else weights_matrix
                struct[layer_index]['weights'] = np.zeros(shape=weights_matrix.shape)
                struct[layer_index]['biases'] = np.zeros(shape=(len(weights_matrix[0, :])))
                layer_index += 1
            else:
                pass
        return struct

    def forward(self, net_input, training):
        """
        Performs a forward pass on the whole Network
        :param net_input: net's input vector/matrix
        :return: net's output vector/matrix
        """
        output = net_input
        if training:
            for layer in self.layers:
                net_input = output
                output = layer.forward_pass(net_input)
            return output
        elif not training:
            for layer in self.dense_configuration:
                net_input = output
                output = self.layers[layer].forward_pass(net_input)
            return output

    def compile(self, optimizer='sgd', loss='squared', metrics='mee', early_stopping=False, patience=3, tolerance=1e-2,
                monitor='loss', mode='growth'):
        """
        Prepares the network for training by assigning an optimizer to it and setting its parameters
        :param opt: ('Optimizer' object)
        :param loss: (str) the type of loss function
        :param metric: (str) the type of metric to track (accuracy etc)
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
        latest_units = 0
        for index, layer in enumerate(self.layers):

            if layer.type == 'dense':
                if index == 0:
                    layer.set_input_shape(self.input_shape)
                    layer.compile()
                    latest_units = layer.n_units
                else:
                    layer.set_input_shape(latest_units)
                    layer.compile()
                    latest_units = layer.n_units

        if isinstance(optimizer, str):
            self.optimizer = optimizers[optimizer]
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer

        self.loss = loss
        self.metrics = metrics
        if early_stopping:
            self.__early_stopping = True
            self.__check_stop = EarlyStopping(monitor=monitor, mode=mode, patience=patience, tolerance=tolerance)

    def fit(self, training_data, training_targets, validation_data=None, epochs=1, batch_size=None, validation_split=0,
            shuffle=False):
        """
        Execute the training of the network
        :param training_data: (numpy ndarray) input training set
        :param training_targets: (numpy ndarray) targets for each input training pattern
        :param val_x: (numpy ndarray) input validation set
        :param val_y: (numpy ndarray) targets for each input validation pattern
        :param batch_size: (integer) the size of the batch
        :param epochs: (integer) number of epochs
        :param val_split: percentage of training data to use as validation data (alternative to val_x and val_y)
        """
        # transform sets to numpy array (if they're not already)
        training_data, training_targets = np.array(training_data), np.array(training_targets)
        self.training_data = training_data
        self.training_targets = training_targets
        self.epochs = epochs
        self.batch_size = batch_size

        if batch_size == 'all':
            batch_size = 1
        if batch_size == None:
            batch_size = 1

        target_len = training_targets.shape[1] if len(training_targets.shape) > 1 else 1
        n_patterns = training_data.shape[0] if len(training_data.shape) > 1 else 1
        n_targets = training_targets.shape[0] if len(training_targets.shape) > 1 else 1

        if target_len != self.__layers[-1].n_units or n_patterns != n_targets or batch_size > n_patterns:
            raise AttributeError(f"Mismatching shapes")

        return self.optimizer.optimization_process(self, train_dataset=training_data, train_labels=training_targets,
                                                   epochs=self.epochs,
                                                   batch_size=self.batch_size, shuffle=shuffle,
                                                   validation=validation_data, early_stopping=self.__early_stopping,
                                                   check_stop=self.__check_stop)

    def predict(self, prediction_input):
        """
        Computes the outputs for a batch of patterns, useful for testing w/ a blind test set
        :param net_input: batch of input patterns
        :return: array of net's outputs
        :param disable_tqdm: (bool) if True disables the progress bar
        """
        prediction_input = np.array(prediction_input)
        prediction_input = prediction_input[np.newaxis, :] if len(prediction_input.shape) < 2 else prediction_input
        predictions = []
        for single_input in (prediction_input):
            predictions.append(self.forward(net_input=single_input, training=False))

        return np.array(predictions)

    def evaluate(self, validation_data, targets, metric=None, loss=None):
        """
        Performs an evaluation of the network based on the targets and either the pre-computed outputs ('net_outputs')
        or the input data ('net_input'), on which the net will first compute the output.
        If both 'predicted' and 'net_input' are None, an AttributeError is raised
        :param targets: the targets for the input on which the net is evaluated
        :param metric: the metric to track for the evaluation
        :param loss: the loss to track for the evaluation
        :param net_outputs: the output of the net for a certain input
        :param net_input: the input on which the net has to be evaluated
        :return: the loss and the metric
        :param disable_tqdm: (bool) if True disables the progress bar
        """
        if metric == None:
            metric = self.metrics
        if loss == None:
            loss = self.loss

        net_outputs = self.predict(validation_data)
        metric_score = np.zeros(self.layers[-1].n_units)
        loss_scores = np.zeros(self.layers[-1].n_units)
        for x, y in zip(net_outputs, targets):
            #metric_score = np.add(metric_score, metrics[metric].function(predicted=x, target=y))
            loss_scores = np.add(loss_scores, losses[loss].function(predicted=x, target=y))
        """ print("----predictions--------")
        print(np.array(net_outputs))
        print("---------------------")
        print("----targets--------")
        print(np.array(targets))
        print("---------------------")
        input() """
        metric_score = metrics[metric].function(predicted=net_outputs,target=targets)
        loss_scores = np.sum(loss_scores) / len(loss_scores)
        #metric_score = np.sum(metric_score) / len(metric_score)
        loss_scores /= len(net_outputs)
        #metric_score /= len(net_outputs)
        return loss_scores, metric_score

    def propagate_back(self, dErr_dOut, gradient_network):
        """
        Propagates back the error to update each layer's gradient
        :param dErr_dOut: derivatives of the error wrt the outputs
        :param grad_net: a structure with the same topology of the neural network in question, but used to store the
        gradients. It will be updated and returned back to the caller
        :return: the updated grad_net
        """
        curr_delta = dErr_dOut
        total_len = 0

        for layer in self.layers:
            if layer.type == 'dense':
                total_len += 1

        for layer_index in reversed(range(total_len)):
            curr_delta, grad_w, grad_b = self.layers[self.dense_configuration[layer_index]].backward_pass(curr_delta)
            gradient_network[layer_index]['weights'] = np.add(gradient_network[layer_index]['weights'], grad_w)
            gradient_network[layer_index]['biases'] = np.add(gradient_network[layer_index]['biases'], grad_b)
        return gradient_network
