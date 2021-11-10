#  Copyright (c) 2021.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import optimizers
from NeuralNetworkCore.Reguralizers import EarlyStopping


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

    def double_split(self, data, labels):
        """
        Split dataset into training and validation with a 65%-35% split.
        :param args:
        :return:
        """

        self.training_set = data[:int(len(data) * 0.65)]
        self.validation_set = data[-int(len(data) * 0.35):]
        self.training_set_labels = labels[:int(len(labels) * 0.65), :]
        self.validation_set_labels = labels[-int(len(labels) * 0.35):, :]

        return [self.training_set, self.training_set_labels], [self.validation_set, self.validation_set_labels]


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
            self.training_set.append(np.concatenate(tmp_test))

        temp_array = np.split(data[1], splits)
        for x in temp_array:
            tmp_test = []
            self.validation_set_labels.append(x)
            for y in temp_array:
                if not np.array_equal(x, y):
                    tmp_test.append(y)
            self.training_set_labels.append(np.concatenate(tmp_test))

        return [self.training_set, self.training_set_labels], [self.validation_set, self.validation_set_labels]


class HyperparametersSearch:
    def __init__(self, name):
        self.__name = name
        self.__best_val = None
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

    @history.setter
    def history(self, history):
        self.__history = history

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
        if model.optimizer != None:
            self.__evaluated_optimizer = model.optimizer
        else:
            self.__evaluated_optimizer, _, _, _ = Model.compile_default()
            self.__evaluated_optimizer = optimizers[self.__evaluated_optimizer]
        self.__optimizer_seen = False
        self.__temp_suspended = {}
        self.__evaluated_optimizer
        _, self.__current_loss, self.__current_metric = Model.compile_default()

        self.__all_reg_mode = False
        self.__reguralizers = {}
        self.__es = False
        self.__monitor, self.__es_mode, self.__patience, self.__tol = EarlyStopping.default()
        self.__epochs = 50
        self.__batch_size = None
        self.__shuffle = False
        self.__cv = 3
        self.__best_val = None
        self.__best_params = None
        self.__best_tr_metric = None
        self.__best_tr_loss = None
        self.__best_val_metric = None

    @property
    def best_params(self):
        return self.__best_params
    @property
    def best_tr_metric(self):
        return self.__best_tr_metric
    @property
    def best_tr_loss(self):
        return self.__best_tr_loss

    @best_tr_metric.setter
    def best_tr_metric(self, best_tr_metric):
        self.__best_tr_metric = best_tr_metric

    @property
    def best_val_metric(self):
        return self.__best_val_metric

    @best_val_metric.setter
    def best_val_metric(self, best_val_metric):
        self.__best_val_metric = best_val_metric

    def add_optimizer_parameters(self, param_combination, param, x):
        if x[0] == 'mom' or x[0] == 'momentum' or x[0] == 'm':
            if self.__optimizer_seen:
                self.__evaluated_optimizer.momentum = float(param_combination[param])
            else:
                self.__temp_suspended[param] = float(param_combination[param])
        if x[0] == 'learningrate' or x[0] == 'lr':
            if self.__optimizer_seen:
                self.__evaluated_optimizer.lr = float(param_combination[param])
            else:
                self.__temp_suspended[param] = float(param_combination[param])
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
                    message = str(self.__evaluated_optimizer().name) + ' has no param ' + param + '.'
                    warnings.warn(message)



            else:
                self.__temp_suspended[param] = param_combination[param]
        if x[0] == 'beta2' or x[0] == 'b2':
            if self.__optimizer_seen:
                if self.__evaluated_optimizer.name == 'adam':
                    self.__evaluated_optimizer.beta2 = param_combination[param]
                else:
                    warnings.warn(str(self.__evaluated_optimizer().name) + ' has no param ' + param + '.',
                                  category='Warning', stacklevel=2)


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

    def fit(self, training_data, training_targets, epochs=None, batch_size=None, shuffle=None, cv=3,
            filename='./curr_dataset'):

        if cv is not None and cv >0:
            splitter = KFold()
            self.__training_set, self.__validation_set = splitter.split((training_data, training_targets), cv)
            self.__cv = cv
        elif cv!=-1:
            splitter = SimpleHoldout()
            self.__training_set, self.__validation_set = splitter.double_split(training_data, training_targets)
            self.__cv = cv
        else:
            self.__cv = cv
            self.__training_set=[training_data, training_targets]



        if epochs == None:
            epochs = self.__epochs
        if batch_size == None:
            batch_size = self.__batch_size
        if batch_size == 'all':
            batch_size = 1

        if shuffle == None:
            shuffle = self.__shuffle
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
        print("EVALUATING " + str(len(experiments)) + ' Combinations for a total of' + str(
            len(experiments) * self.__cv) + ' times.')
        f = open('./results/results_' + filename + '.txt', 'w+')
        f.writelines("CUP GRIDSEARCH RESULTS")
        f.write('\n')
        f.writelines('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        f.write('\n')
        f.writelines(self.__param_list)
        f.write('\n')
        f.writelines('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        f.write('\n')
        for outmost_index, param_combination in enumerate(experiments):
            print(param_combination)
            self.__optimizer_seen = False
            self.__reguralizers = {}

            for param in param_combination:
                x = param.split('_')
                if x[0] == 'epochs':
                    self.__epochs = param_combination[param]
                if x[0] == 'batchsize':
                    self.__batch_size = param_combination[param]
                if x[0] == 'shuffle':
                    self.__shuffle = param_combination[param]
                if x[0] == 'units':
                    index = self.__model.dense_configuration[int(x[1]) - 1]
                    self.__model.layers[index].n_units = param_combination[param]
                if x[0] == 'dropouts' or x[0] == 'drop' or x[0] == 'drops':
                    if (len(x) != 2):
                        raise AttributeError("Dropout layer not specified")
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
                        for suspended_parameter in self.__temp_suspended:
                            self.add_optimizer_parameters(self.__temp_suspended, suspended_parameter,
                                                          [suspended_parameter])

                if x[0] == 'loss' or x[0] == 'losses':
                    self.__current_loss = param_combination[param]
                if x[0] == 'metric' or x[0] == 'metrics':
                    self.__current_metric = param_combination[param]
                if x[0] == 'regularizer' or x[0] == 'reg':
                    if x[1] == 'all':
                        for denselayer in self.model.dense_configuration:
                            self.__model.layers[denselayer].regularizer = param_combination[param]
                        self.__all_reg_mode = True
                    else:
                        if x[1] in self.__reguralizers:
                            self.__reguralizers[x[1]]['type'] = param_combination[param]
                        else:
                            self.__reguralizers[x[1]] = {'type': param_combination[param], 'value': -1}
                if x[0] == 'regularizerparam' or x[0] == 'regparam':
                    if x[1] in self.__reguralizers:
                        self.__reguralizers[x[1]]['value'] = param_combination[param]
                    else:
                        self.__reguralizers[x[1]] = {'type': None, 'value': param_combination[param]}
                if x[0] == 'weightinit' or x[0] == 'winit' or x[0] == 'weightinitializer':
                    index = self.__model.dense_configuration[int(x[1]) - 1]
                    self.__model.layers[index].weight_initializer = param_combination[param]
                if x[0] == 'biasinit' or x[0] == 'binit' or x[0] == 'biasinitializier':
                    index = self.__model.dense_configuration[int(x[1]) - 1]
                    self.__model.layers[index].bias_initializer = param_combination[param]
                if x[0] == 'act' or x[0] == 'actfun' or x[0] == 'activationfunction':
                    index = self.__model.dense_configuration[int(x[1]) - 1]
                    self.__model.layers[index].activation_function = param_combination[param]
                if x[0] == 'earlystopping' or x[0] == 'es':
                    self.__es = param_combination[param]
                if x[0] == 'monitor':
                    self.__monitor = param_combination[param]
                if x[0] == 'mode':
                    self.__es_mode = param_combination[param]
                if x[0] == 'patience':
                    self.__patience = param_combination[param]
                if x[0] == 'tol' or x[0] == 'tolerance' or x[0] == 'tollerance':
                    self.__tol = param_combination[param]

            for reg in self.__reguralizers:
                if self.__model.layers[self.__model.dense_configuration[int(reg) - 1]].regularizer != None:
                    self.__reguralizers[reg]['type'] = self.__model.layers[
                        self.__model.dense_configuration[int(reg) - 1]].regularizer
                if self.__reguralizers[reg]['type'] == None or self.__reguralizers[reg]['value'] == -1:
                    warnings.warn("Mismatching regularizers and params, skipping ")
                else:
                    self.__model.layers[self.__model.dense_configuration[int(reg) - 1]].regularizer = \
                        self.__reguralizers[reg]['type']
                    self.__model.layers[self.__model.dense_configuration[int(reg) - 1]].regularizer_param = \
                        self.__reguralizers[reg]['value']
            self.__model.compile(optimizer=self.__evaluated_optimizer, loss=self.__current_loss,
                                 metrics=self.__current_metric, early_stopping=self.__es, patience=self.__patience,
                                 tolerance=self.__tol, monitor=self.__monitor, mode=self.__es_mode)
            self.__model.showLayers()
            results = {}
            if cv is not None and cv > 0:
                for index, training_set in enumerate(self.__training_set[0]):
                    print('Fold[' + str(index + 1) + ']')


                    res = self.__model.fit(training_set, self.__training_set[1][index],
                                           validation_data=(
                                               self.__validation_set[0][index], self.__validation_set[1][index]),
                                           epochs=self.__epochs,
                                           batch_size=self.__batch_size, shuffle=self.__shuffle)

                    if index == 0:
                        results['training_error'] = np.asarray(res['training_error'])
                        results['training_metrics'] = np.asarray(res['training_metrics'])
                        results['validation_error'] = np.asarray(res['validation_error'])
                        results['validation_metrics'] = np.asarray(res['validation_metrics'])
                    else:

                        results['training_error'] = np.add(results['training_error'], np.asarray(res['training_error']))

                        results['training_metrics'] = np.add(results['training_metrics'],
                                                             np.asarray(res['training_metrics']))
                        results['validation_error'] = np.add(results['validation_error'],
                                                             np.asarray(res['validation_error']))
                        results['validation_metrics'] = np.add(results['validation_metrics'],
                                                               np.asarray(res['validation_metrics']))
                results['training_error'] = np.divide(results['training_error'], self.__cv)
                results['training_metrics'] = np.divide(results['training_metrics'], self.__cv)
                results['validation_error'] = np.divide(results['validation_error'], self.__cv)
                results['validation_metrics'] = np.divide(results['validation_metrics'], self.__cv)

                if self.__best_val is None:
                    self.__best_val = results['validation_metrics'][-1]
                    self.best_model = self.__model
                    self.__best_params = param_combination
                    self.__best_tr_metric = results['training_metrics'][-1]
                    self.__best_tr_loss = results['training_error'][-1]
                    self.__best_val_metric = results['validation_metrics'][-1]
                if self.__current_metric == 'mee':
                    if self.__best_val > results['validation_metrics'][-1]:
                        self.__best_val = results['validation_metrics'][-1]
                        self.__best_tr_metric = results['training_metrics'][-1]
                        self.__best_val_metric = results['validation_metrics'][-1]
                        self.__best_tr_loss = results['training_error'][-1]
                        self.best_model = self.__model
                        self.__best_params = param_combination
                elif self.__current_metric == 'binary':
                    if self.__best_val < results['validation_metrics'][-1]:
                        self.__best_val = results['validation_metrics'][-1]
                        self.__best_tr_metric = results['training_metrics'][-1]
                        self.__best_val_metric = results['validation_metrics'][-1]
                        self.__best_tr_loss = results['training_error'][-1]
                        self.best_model = self.__model
                        self.__best_params = param_combination
            elif self.__cv!=-1:

                res = self.__model.fit(self.__training_set[0], self.__training_set[1],
                                       validation_data=(
                                           self.__validation_set[0], self.__validation_set[1]),
                                       epochs=self.__epochs,
                                       batch_size=self.__batch_size, shuffle=self.__shuffle)

                results['training_error'] = res['training_error']
                results['training_metrics'] = res['training_metrics']
                results['validation_error'] = res['validation_error']
                results['validation_metrics'] = res['validation_metrics']

                if self.__best_val is None:
                    self.__best_val = results['validation_metrics'][-1]
                    self.best_model = self.__model
                    self.__best_params = param_combination
                    self.__best_tr_metric = results['training_metrics'][-1]
                    self.__best_val_metric = results['validation_metrics'][-1]
                    self.__best_tr_loss = results['training_error'][-1]
                if self.__current_metric == 'mee':
                    if self.__best_val > results['validation_metrics'][-1]:
                        self.__best_val = results['validation_metrics'][-1]
                        self.__best_tr_metric = results['training_metrics'][-1]
                        self.__best_val_metric = results['validation_metrics'][-1]
                        self.__best_tr_loss = results['training_error'][-1]
                        self.best_model = self.__model
                        self.__best_params = param_combination
                elif self.__current_metric == 'binary':
                    if self.__best_val < results['validation_metrics'][-1]:
                        self.__best_val = results['validation_metrics'][-1]
                        self.__best_tr_metric = results['training_metrics'][-1]
                        self.__best_val_metric = results['validation_metrics'][-1]
                        self.best_model = self.__model
                        self.__best_tr_loss = results['training_error'][-1]
                        self.__best_params = param_combination
            else:
                print(self.__model.optimizer.lr)
                print(self.__model.optimizer.momentum)
                res = self.__model.fit(self.__training_set[0], self.__training_set[1],
                                       epochs=self.__epochs,
                                       batch_size=self.__batch_size, shuffle=self.__shuffle)

                results['training_error'] = res['training_error']
                results['training_metrics'] = res['training_metrics']


                if self.__best_val is None:
                    self.__best_val = results['training_metrics'][-1]
                    self.best_model = self.__model
                    self.__best_params = param_combination
                    self.__best_tr_metric = results['training_metrics'][-1]
                    self.__best_tr_loss = results['training_error'][-1]
                if self.__current_metric == 'mee':
                    if self.__best_val > results['training_metrics'][-1]:
                        self.__best_tr_metric = results['training_metrics'][-1]
                        self.best_model = self.__model
                        self.__best_params = param_combination
                        self.__best_tr_loss = results['training_error'][-1]
                elif self.__current_metric == 'binary':
                    if self.__best_val < results['training_metrics'][-1]:
                        self.__best_val = results['training_metrics'][-1]
                        self.__best_tr_metric = results['training_metrics'][-1]
                        self.best_model = self.__model
                        self.__best_tr_loss = results['training_error'][-1]
                        self.__best_params = param_combination


            f.writelines('________param_combination________')
            f.write('\n')
            for key, value in param_combination.items():
                f.write('%s:%s\t' % (key, value))
            f.write('\n')
            f.writelines('________index__________________')
            f.write('\n')
            f.write(str(outmost_index))
            f.write('\n')
            f.writelines('________Training_error________')
            f.write('\n')
            f.writelines(str(results['training_error'][-1]))
            f.write('\n')

            if self.__cv!=-1:
                f.writelines('________Validation_error________')
                f.write('\n')
                f.writelines(str(results['validation_error'][-1]))
                f.write('\n')
            f.writelines('________Training_metrics________')
            f.write('\n')
            f.writelines(str(results['training_metrics'][-1]))
            f.write('\n')
            if self.__cv != -1:
                f.writelines('________Validation_metrics________')
                f.write('\n')
                f.writelines(str(results['validation_metrics'][-1]))
                f.write('\n')
            f.writelines("___________________________________")
            f.write('\n')
            if self.__cv != -1:
                self.history[outmost_index] = {'results': results, 'parameters': param_combination}
            plt.figure()
            plt.title("LOSS")
            plt.plot(results['training_error'])
            if self.__cv != -1:
                plt.plot(results['validation_error'])
                plt.legend(['training', 'validation'])
            try:
                os.makedirs('./plots/' + filename)
            except FileExistsError:
                pass

            plt.savefig('./plots/' + filename + '/' + (str(outmost_index)) + 'error.png')
            plt.figure()
            plt.title("METRICS")
            plt.plot(results['training_metrics'])
            if self.__cv != -1:
                plt.plot(results['validation_metrics'])
                plt.legend(['training', 'validation'])
            plt.savefig('./plots/' + filename + '/' + (str(outmost_index)) + 'metrics.png')
        f.writelines("BEST RESULTS ")
        f.write('\n')
        if self.__cv != -1:
            f.writelines('Validation metrics')
            f.write('\n')
            f.write(str(self.__best_val))
            f.write('\n')
        for key, value in self.__best_params.items():
            f.write('%s:%s\t' % (key, value))
        f.close()
