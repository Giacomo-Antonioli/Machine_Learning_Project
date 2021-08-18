#  Copyright (c) 2021.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import itertools
import warnings
import sys
warnings.simplefilter('always', UserWarning)
import numpy as np

from NeuralNetworkCore.Optimizers import optimizers


# def custom_formatwarning(msg, *args, **kwargs):
#     # ignore everything except the message
#     return str(msg) + '\n'
#
# warnings.formatwarning = custom_formatwarning


class ValidationTechnique:
    def __init__(self, name):
        self.__name = name
        self.__training_set = []
        self.__validation_set = []
        self.__test_set = []
        self.__training_set_labels = []
        self.__validation_set_labels = []
        self.__test_set_labels = []

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, training_set):
        self.__training_set = training_set

    @property
    def validation_set(self):
        return self.__validation_set

    @validation_set.setter
    def validation_set(self, validation_set):
        self.__validation_set = validation_set

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, test_set):
        self.__test_set = test_set

    @property
    def training_set_labels(self):
        return self.__training_set_labels

    @training_set_labels.setter
    def training_set_labels(self, training_set_labels):
        self.__training_set_labels = training_set_labels

    @property
    def validation_set_labels(self):
        return self.__validation_set_labels

    @validation_set_labels.setter
    def validation_set_labels(self, validation_set_labels):
        self.__validation_set_labels = validation_set_labels

    @property
    def test_set_labels(self):
        return self.__test_set_labels

    @test_set_labels.setter
    def test_set_labels(self, test_set_labels):
        self.__test_set_labels = test_set_labels


class SimpleHoldout(ValidationTechnique):

    def __init__(self):
        super().__init__('Simple_holdout')

    def split(self, *args):
        """
        Split data in 3 parts (50% training, 25% validation, 25% test-set or with a percentage used defined)
        :param args:
        :return:
        """

        if len(args) == 1:

            data = args[0][0]
            labels = args[0][1]
            self.training_set = data[:int(len(data) * 0.5)]  # get first 50% of file list
            self.validation_set = data[int(len(data) * 0.5):int(len(data) * 0.75)]  # get middle 25% of file list
            self.test_set = data[-int(len(data) * 0.25):]  # get last 25% of file list
            self.training_set_labels = labels[:int(len(labels) * 0.5)]  # get first 50% of file list
            self.validation_set_labels = labels[
                                         int(len(labels) * 0.5):int(len(data) * 0.75)]  # get middle 25% of file list
            self.test_set_labels = labels[-int(len(labels) * 0.25):]  # get last 25% of file list
            return [self.training_set, self.training_set_labels], [self.validation_set, self.validation_set_labels], [
                self.test_set, self.test_set_labels]

        elif len(args) == 2:

            data = args[0][0]
            labels = args[0][1]
            self.training_set = data[:int(len(data) * args[1][0])]  # get first 50% of file list
            self.validation_set = data[int(len(data) * args[1][0]):int(len(data) * args[1][0]) + int(
                len(data) * args[1][1])]  # get middle 25% of file list
            self.test_set = data[-int(len(data) * args[1][2]):]  # get last 25% of file list
            self.training_set_labels = labels[:int(len(labels) * args[1][0])]  # get first 50% of file list
            self.validation_set_labels = labels[int(len(labels) * args[1][0]):int(len(labels) * args[1][0]) + int(
                len(labels) * args[1][1])]  # get middle 25% of file list
            self.test_set_labels = labels[-int(len(labels) * args[1][2]):]  # get last 25% of file list
            return [self.training_set, self.training_set_labels], [self.validation_set, self.validation_set_labels], [
                self.test_set, self.test_set_labels]
        else:
            print("wrong usage of the function")

    def double_split(self, *args):
        """
        Split dataset into training and validation with a 65%-35% split.
        :param args:
        :return:
        """
        data = args[0][0]
        labels = args[0][1]
        self.training_set = data[:int(len(data) * 0.65)]  # get first 50% of file list
        self.validation_set = data[-int(len(data) * 0.35):]  # get middle 25% of file list
        self.training_set_labels = labels[:int(len(labels) * 0.65)]  # get first 50% of file list
        self.validation_set_labels = labels[-int(len(labels) * 0.35)]  # get middle 25% of file list
        return [self.training_set, self.training_set_labels], [self.validation_set, self.validation_set_labels], [
            self.test_set, self.test_set_labels]


class KFold(ValidationTechnique):
    def __init__(self):
        super().__init__('KFold Technique')

    def split(self, data, splits=5):
        temp_array = np.split(data[0], splits)
        for x in temp_array:
            tmp_test = []
            self.validation_set.append(x)
            for y in temp_array:
                if not np.array_equal(x, y):
                    tmp_test.append(y)
            self.training_set.append(tmp_test)
        temp_array = np.split(data[1], splits)
        for x in temp_array:
            tmp_test = []
            self.validation_set_labels.append(x)
            for y in temp_array:
                if not np.array_equal(x, y):
                    tmp_test.append(y)
            self.training_set_labels.append(tmp_test)
        return [self.training_set, self.training_set_labels], [self.validation_set, self.validation_set_labels]


class DoubleCrossValidation(ValidationTechnique):
    def __init__(self):
        super().__init__('Double cross validation')
        self.__outer_holdout_splitter = SimpleHoldout()
        self.__inner_kFold_splitter = KFold()

    @property
    def outer_houldout_splitter(self):
        return self.__outer_houldout_splitter

    @outer_houldout_splitter.setter
    def outer_houldout_splitter(self, outer_houldout_splitter):
        self.__outer_houldout_splitter = outer_houldout_splitter

    @property
    def inner_kFold_splitter(self):
        return self.__inner_kFold_splitter

    @inner_kFold_splitter.setter
    def inner_kFold_splitter(self, inner_kFold_splitter):
        self.__inner_kFold_splitter = inner_kFold_splitter

    def split(self, data, splits=5):
        tmp_test, tmp_val, tmp_test = self.__outer_houldout_splitter(data)


class HyperparametersSearch:
    def __init__(self, name):
        self.__name = name
        self.__best_parameters = []
        self.__history = {}
        self.__best_model = None

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def best_parameters(self):
        return self.__best_parameters

    @best_parameters.setter
    def best_parameters(self, best_parameters):
        self.__best_parameters = best_parameters

    @property
    def history(self):
        return self.__history

    @property
    def best_model(self):
        return self.__best_model

    @best_model.setter
    def best_model(self, best_model):
        self.__best_model = best_model


class GridSearch(HyperparametersSearch):

    def __init__(self, model, param_list):
        super().__init__('GridSearch')
        self.__model = model
        self.__param_list = param_list
        self.__training_set = []
        self.__validation_set = []
        self.__optimizer_seen = False
        self.__temp_suspended = {}
        self.__evaluated_optimizer = []

    def add_optimizer_parameters(self, param_combination, param, x):
        if x[0] == 'mom' or x[0] == 'momentum' or x[0] == 'm':
            if self.__optimizer_seen:
                self.__evaluated_optimizer.momentum = param_combination[param]
            else:
                self.__temp_suspended[param] = param_combination[param]
        if x[0] == 'learningrate' or x[0] == 'lr':
            if self.__optimizer_seen:
                self.__evaluated_optimizer.lr = param_combination[param]
            else:
                self.__temp_suspended[param] = param_combination[param]
        if x[0] == 'nesterov':
            if self.__optimizer_seen:
                if self.__evaluated_optimizer.name == 'sgd':
                    self.__evaluated_optimizer.momentum = param_combination[param]
                else:
                    warnings.warn(str(self.__evaluated_optimizer().name) + ' has no param ' + param + '.')

            else:
                self.__temp_suspended[param] = param_combination[param]
        if x[0] == 'rho':
            if self.__optimizer_seen:
                if self.__evaluated_optimizer.name == 'rmsprop':
                    self.__evaluated_optimizer.rho = param_combination[param]
                else:
                    warnings.warn(str(self.__evaluated_optimizer().name) + ' has no param ' + param + '.')

            else:
                self.__temp_suspended[param] = param_combination[param]
        if x[0] == 'beta1' or x[0] == 'b1':
            if self.__optimizer_seen:
                if self.__evaluated_optimizer.name == 'adam':
                    self.__evaluated_optimizer.beta1 = param_combination[param]
                else:
                    message=str(self.__evaluated_optimizer().name) + ' has no param ' + param + '.'
                    warnings.warn(message)
                    sys.stderr.flush()


            else:
                self.__temp_suspended[param] = param_combination[param]
        if x[0] == 'beta2' or x[0] == 'b2':
            if self.__optimizer_seen:
                if self.__evaluated_optimizer.name == 'adam':
                    self.__evaluated_optimizer.beta2 = param_combination[param]
                else:
                    warnings.warn(str(self.__evaluated_optimizer().name) + ' has no param ' + param + '.')

            else:
                self.__temp_suspended[param] = param_combination[param]
        if x[0] == 'epsilon' or x[0] == 'e':
            if self.__optimizer_seen:
                if self.__evaluated_optimizer.name == 'adam':
                    self.__evaluated_optimizer.epsilon = param_combination[param]
                else:
                    warnings.warn(str(self.__evaluated_optimizer().name) + ' has no param ' + param + '.')

            else:
                self.__temp_suspended[param] = param_combination[param]

    def fit(self, training_data, training_targets, epochs=1, batch_size=None, shuffle=False, cv=3):
        if cv is not None or cv != 0:
            splitter = KFold()
            self.__training_set, self.__validation_set = splitter.split((training_data, training_targets), cv)
        else:
            splitter = SimpleHoldout()
            self.__training_set, self.__validation_set = splitter.double_split((training_data, training_targets))

        if isinstance(epochs, int):
            self.__param_list['epochs'] = [epochs]
        else:
            self.__param_list['epochs'] = epochs
        if isinstance(batch_size, int):
            self.__param_list['batchsize'] = [batch_size]
        else:
            self.__param_list['batchsize'] = batch_size

        keys, values = zip(*self.__param_list.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for param_combination in experiments:
            print(param_combination)
            self.__optimizer_seen = False

            for param in param_combination:
                x = param.split('_')
                if x[0] == 'units':
                    index = self.__model.dense_configuration[int(x[1]) - 1]
                    self.__model.layers[index].n_units = param_combination[param]
                if x[0] == 'dropouts' or x[0] == 'drop' or x[0] == 'drops':
                    counter = int(x[1])
                    index = 0
                    for layer in self.__model.layers:
                        if layer.type == 'drop' and counter != 0:
                            index += 1
                            counter -= 1

                    if index != 0:
                        self.__model.layers[index].probability = param_combination[param]

                self.add_optimizer_parameters(param_combination, param, x)

                if x[0] == 'optimizers' or x[0] == 'optimizer' or x[0] == 'opt':
                    self.__optimizer_seen = True
                    self.__evaluated_optimizer = optimizers[param_combination[param]]
                    if len(self.__temp_suspended) > 0:
                        print(self.__temp_suspended)
                        for suspended_parameter in self.__temp_suspended:
                            self.add_optimizer_parameters(self.__temp_suspended, suspended_parameter,
                                                          [suspended_parameter])
                    print(self.__evaluated_optimizer)
                    # for delayed_params in self.__temp_suspended:
                    #loss
                    #metrics
                    #fit
                    #save results
            self.__model.showLayers()

        # for params
        # for split
        # check params
        # fit
        # save results
