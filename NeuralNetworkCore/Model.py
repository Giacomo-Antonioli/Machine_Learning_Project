from NeuralNetworkCore.Optimizers import *


class Model:
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
        self.__net_configuration_types=[]

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

    def add(self, object):
        self.layers.append(object)
        if object.type == 'dense':
            self.dense_configuration.append(self.get_layer_depth())
        self.net_configuration_types.append(object.type)

    def get_layer_depth(self):

        return len(self.layers)-1


    def showLayers(self):
        print("°°°°°°°°°°°°°°°°°°°°°")
        for x in self.__layers:
            if x.type!='drop':
                print(x.n_units)

        print("°°°°°°°°°°°°°°°°°°°°°")

    def get_empty_struct(self):
        """ :return: a zeroed structure with the same topology of the NN to contain all the layers' gradients """
        valid_layers = 0
        for layer in self.layers:
            if layer.type == 'dense':
                valid_layers += 1
        struct = np.array([{}] * valid_layers)
        layer_index = 0
        for index,layer in enumerate(self.layers):
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

    def forward(self, net_input):
        """
        Performs a forward pass on the whole Network
        :param net_input: net's input vector/matrix
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
        if isinstance(optimizer, str):
            self.optimizer = optimizers[optimizer]
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer

        self.loss = loss
        self.metrics = metrics

        # self.__params = {**self.__params, **{'loss': loss, 'metr': metr, 'lr': lr, 'lr_decay': lr_decay,
        #                                      'limit_step': limit_step, 'decay_rate': decay_rate,
        #                                      'decay_steps': decay_steps, 'staircase': staircase, 'momentum': momentum,
        #                                      'reg_type': reg_type, 'lambd': lambd}}
        # self.__opt = optimizers[opt](net=self, loss=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
        #                              decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase,
        #                              momentum=momentum, reg_type=reg_type, lambd=lambd)

    def fit(self, training_data, training_targets, validation_data=None, epochs=1, batch_size=None, validation_split=0,
            shuffle=False, **kwargs):
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
        # use validation data
        # if val_x is not None and val_y is not None:
        #     if val_split != 0:
        #         warnings.warn(f"A validation split was given, but instead val_x and val_y will be used")
        #     val_x, val_y = np.array(val_x), np.array(val_y)
        #     n_patterns = val_x.shape[0] if len(val_x.shape) > 1 else 1
        #     n_targets = val_y.shape[0] if len(val_y.shape) > 1 else 1
        #     if n_patterns != n_targets:
        #         raise AttributeError(f"Mismatching shapes {n_patterns} {n_targets}")
        # else:
        #     # use validation split
        #     if val_split != 0:
        #         if val_split < 0 or val_split > 1:
        #             raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
        #         indexes = np.random.randint(low=0, high=len(training_data), size=math.floor(val_split * len(training_data)))
        #         val_x = training_data[indexes]
        #         val_y = training_targets[indexes]
        #         training_data = np.delete(training_data, indexes, axis=0)
        #         training_targets = np.delete(training_targets, indexes, axis=0)

        # check that the shape of the target matches the net's architecture
        if batch_size == None:
            batch_size = len(training_data)
            print("true")
        target_len = training_targets.shape[1] if len(training_targets.shape) > 1 else 1
        print(target_len)
        print(training_targets.shape)
        n_patterns = training_data.shape[0] if len(training_data.shape) > 1 else 1
        print(n_patterns)
        n_targets = training_targets.shape[0] if len(training_targets.shape) > 1 else 1
        print(n_targets)
        if target_len != self.__layers[-1].n_units or n_patterns != n_targets or batch_size > n_patterns:
            raise AttributeError(f"Mismatching shapes")

        return self.optimizer.optimization_process(self, training_data, training_targets, epochs=self.epochs,
                                                   batch_size=self.batch_size, shuffle=shuffle,
                                                   validation=validation_data)

    def predict(self, prediction_input, disable_tqdm=True):
        """
        Computes the outputs for a batch of patterns, useful for testing w/ a blind test set
        :param net_input: batch of input patterns
        :return: array of net's outputs
        :param disable_tqdm: (bool) if True disables the progress bar
        """
        prediction_input = np.array(prediction_input)
        prediction_input = prediction_input[np.newaxis, :] if len(prediction_input.shape) < 2 else prediction_input
        predictions = []
        for single_input in tqdm.tqdm(prediction_input, desc="Predicting patterns", disable=disable_tqdm):
            predictions.append(self.forward(net_input=single_input))
        return np.array(predictions)

    def evaluate(self, targets, metr, loss, net_outputs=None, net_input=None, disable_tqdm=True):
        """
        Performs an evaluation of the network based on the targets and either the pre-computed outputs ('net_outputs')
        or the input data ('net_input'), on which the net will first compute the output.
        If both 'predicted' and 'net_input' are None, an AttributeError is raised
        :param targets: the targets for the input on which the net is evaluated
        :param metr: the metric to track for the evaluation
        :param loss: the loss to track for the evaluation
        :param net_outputs: the output of the net for a certain input
        :param net_input: the input on which the net has to be evaluated
        :return: the loss and the metric
        :param disable_tqdm: (bool) if True disables the progress bar
        """
        if net_outputs is None:
            if net_input is None:
                raise AttributeError("Both net_outputs and net_input cannot be None")
            net_outputs = self.predict(net_input, disable_tqdm=disable_tqdm)
        metr_scores = np.zeros(self.layers[-1].n_units)
        loss_scores = np.zeros(self.layers[-1].n_units)
        for x, y in tqdm.tqdm(zip(net_outputs, targets), total=len(targets), desc="Evaluating model",
                              disable=disable_tqdm):
            metr_scores = np.add(metr_scores, metrics[metr].func(predicted=x, target=y))
            loss_scores = np.add(loss_scores, losses[loss].func(predicted=x, target=y))
        loss_scores = np.sum(loss_scores) / len(loss_scores)
        metr_scores = np.sum(metr_scores) / len(metr_scores)
        loss_scores /= len(net_outputs)
        metr_scores /= len(net_outputs)
        return loss_scores, metr_scores

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
