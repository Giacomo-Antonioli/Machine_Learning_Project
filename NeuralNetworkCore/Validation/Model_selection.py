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
import multiprocessing
from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import optimizers, optimizers_attributes
from NeuralNetworkCore.Reguralizers import EarlyStopping

from multiprocessing import Process, Manager, Pool
os.environ['WANDB_NAME']= 'Machine_Learning_Project'
os.environ['WANDB_API_KEY']= 'local-94c8ff41420f1a793c98053287704ca383313390'
import wandb
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)
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

    def proportional_holdout(self, *args):
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

    def custom_proportions_holdout(self, *args):
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

    def split(self, *args):
        """
        Split data in 3 parts (50% training, 25% validation, 25% test-set or with a percentage used defined)
        :param args:
        :return:
        """

        if len(args) == 1:
            return self.proportional_holdout(args)


        elif len(args) == 2:
            return self.custom_proportions_holdout(args)

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
        temp_array = np.array_split(data[0], splits)
        for x in temp_array:
            tmp_test = []
            self.validation_set.append(x)
            for y in temp_array:
                if not np.array_equal(x, y):
                    tmp_test.append(y)
            self.training_set.append(np.concatenate(tmp_test))

        temp_array = np.array_split(data[1], splits)
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
        self.__optimizers_attrs = optimizers_attributes
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
        self.__pool_size=4
        self.__look_up_dict = {
            **dict.fromkeys(['epochs'], 'epochs'),
            **dict.fromkeys(['batchsize', 'bs'], 'batch_size'),
            **dict.fromkeys(['shuffle'], 'shuffle'),
            **dict.fromkeys(['units', 'unit'], 'n_units'),
            **dict.fromkeys(['drop', 'dropout', 'dropouts', 'drops'], 'drop'),
            **dict.fromkeys(['metric', 'metrics'], 'metric'),
            **dict.fromkeys(['optimizers', 'optimizer', 'opt'], 'optimizers'),
            **dict.fromkeys(['lr', 'learningrate'], 'lr'),
            **dict.fromkeys(['momentum', 'mom', 'm'], 'momentum'),
            **dict.fromkeys(['nesterov'], 'nesterov'),
            **dict.fromkeys(['rho'], 'rho'),
            **dict.fromkeys(['beta1', 'b1'], 'beta1'),
            **dict.fromkeys(['beta2', 'b2'], 'beta2'),
            **dict.fromkeys(['epsilon', 'e'], 'epsilon'),
            **dict.fromkeys(['loss', 'losses'], 'loss_function'),
            **dict.fromkeys(['regularizerparam', 'regparam'], 'regularizer_param'),
            **dict.fromkeys(['regularizer', 'reg'], 'regularizer'),
            **dict.fromkeys(['weightinit', 'winit', 'weightinitializer'], 'weight_initializer'),
            **dict.fromkeys(['biasinit', 'binit', 'biasinitializier'], 'bias_initializer'),
            **dict.fromkeys(['act', 'actfun', 'activationfunction'], 'activation_function'),
            **dict.fromkeys(['earlystopping', 'es'], 'es'),
            **dict.fromkeys(['monitor'], 'monitor'),
            **dict.fromkeys(['mode'], 'es_mode'),
            **dict.fromkeys(['patience'], 'patience'),
            **dict.fromkeys(['tol', 'tollerance', 'tolerance'], 'tol')
        }
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
        self.results = {}

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

    def parameters_skimming(self):
        parameters_new = {}
        try:  ## Control if parameters list has to search over optimizers
            param_optmiziers = self.__param_list['opt']
            print(param_optmiziers)

            for opti in param_optmiziers:  ## if so for each optimizer select attributes that are significant to him
                if isinstance(opti, str):  # otherwise remove them from his cycles to avoid repeteade cycles
                    selected_opti = optimizers[opti]
                    current_key = opti
                    parameters_new[current_key] = {}
                else:
                    selected_opti = opti
                    current_key = get_key(optimizers, opti)
                    parameters_new[current_key] = {}
                print(current_key)
                for key in self.__param_list:
                    if key != 'opt':
                        print("\tsearching key: " + key)
                        try:
                            print(self.__look_up_dict[key])
                            if self.__look_up_dict[key] in optimizers_attributes:
                                if hasattr(selected_opti, self.__look_up_dict[key]):

                                    print("\t\t" + key + " found")

                                    if not key in parameters_new[current_key]:
                                        parameters_new[current_key][key] = []
                                    parameters_new[current_key][key] = (self.__param_list[key])
                            else:
                                if not key in parameters_new[current_key]:
                                    parameters_new[current_key][key] = []
                                parameters_new[current_key][key] = (self.__param_list[key])

                        except AttributeError:
                            print(AttributeError)

        except AttributeError:
            print(AttributeError)

        experimets = []
        for opts in parameters_new:
            curr_dict = parameters_new[opts]
            curr_dict['opt'] = [opts]
            keys, values = zip(*curr_dict.items())
            partial = [dict(zip(keys, v)) for v in itertools.product(*values)]
            experimets = experimets + partial
        return experimets

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

    def generate_current_experiment(self, param_combination):
        for param in param_combination:
            x = param.split('_')
            if x[0] == 'epochs':
                self.__epochs = param_combination[param]
            if x[0] == 'batchsize' or x[0] == 'bs':
                self.__batch_size = param_combination[param]
            if x[0] == 'shuffle':
                self.__shuffle = param_combination[param]
            if x[0] == 'units' or x[0] == 'unit':
                index = self.__model.dense_configuration[int(x[1]) - 1]
                self.__model.layers[index].n_units = param_combination[param]
            if x[0] == 'dropouts' or x[0] == 'dropout' or x[0] == 'drop' or x[0] == 'drops':
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

        print("__________-")

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

    def reset_results(self):
        self.results = {}

    def set_results(self, res):
        self.results['training_error'] = np.asarray(res['training_error'])
        self.results['training_metrics'] = np.asarray(res['training_metrics'])
        self.results['validation_error'] = np.asarray(res['validation_error'])
        self.results['validation_metrics'] = np.asarray(res['validation_metrics'])

    def accumulate_results(self, res):
        self.results['training_error'] = np.add(self.results['training_error'], np.asarray(res['training_error']))

        self.results['training_metrics'] = np.add(self.results['training_metrics'],
                                                  np.asarray(res['training_metrics']))
        self.results['validation_error'] = np.add(self.results['validation_error'],
                                                  np.asarray(res['validation_error']))
        self.results['validation_metrics'] = np.add(self.results['validation_metrics'],
                                                    np.asarray(res['validation_metrics']))

    def get_mean_error(self):
        self.results['training_error'] = np.divide(self.results['training_error'], self.__cv)
        self.results['training_metrics'] = np.divide(self.results['training_metrics'], self.__cv)
        self.results['validation_error'] = np.divide(self.results['validation_error'], self.__cv)
        self.results['validation_metrics'] = np.divide(self.results['validation_metrics'], self.__cv)

    def update_best(self, param_combination):
        self.__best_val = self.results['validation_metrics'][-1]
        self.__best_tr_metric = self.results['training_metrics'][-1]
        self.__best_val_metric = self.results['validation_metrics'][-1]
        self.__best_tr_loss = self.results['training_error'][-1]
        self.best_model = self.__model
        self.__best_params = param_combination

    def update_best_results(self,param_combination):
        if self.__best_val is None:
            self.update_best(param_combination)
        if self.__current_metric == 'mee':
            if self.__best_val > self.results['validation_metrics'][-1]:
                self.update_best(param_combination)
        elif self.__current_metric == 'binary':
            if self.__best_val < self.results['validation_metrics'][-1]:
                self.update_best(param_combination)

    def internal_runs(self, args):

        experiments = args[0]
        cv = args[1]
        print("########################################################################################")
        print("\t\t\t" + str(cv))
        for outmost_index, param_combination in enumerate(experiments):
            config=param_combination
            wandb.init(
                # Set entity to specify your username or team name
                # ex: entity="carey",
                # Set the project where this run will be logged
                project="test" + self.__model.name,
                group="experiment_" + self.__model.name,
                # Track hyperparameters and run metadata
                config=config)
            print(param_combination)
            self.__optimizer_seen = False
            self.__reguralizers = {}

            self.generate_current_experiment(param_combination)

            self.__model.compile(optimizer=self.__evaluated_optimizer, loss=self.__current_loss,
                                 metrics=self.__current_metric, early_stopping=self.__es, patience=self.__patience,
                                 tolerance=self.__tol, monitor=self.__monitor, mode=self.__es_mode)
            #self.__model.showLayers()

            self.reset_results()
            if cv is not None and cv > 0:
                for index, training_set in enumerate(self.__training_set[0]):
                    print('Fold[' + str(index + 1) + ']')
                    print('trainingSet: '+str(len(training_set)))
                    print('epochs: '+str(self.__epochs))
                    res = self.__model.fit(training_set, self.__training_set[1][index],
                                           validation_data=(
                                               self.__validation_set[0][index], self.__validation_set[1][index]),
                                           epochs=self.__epochs,
                                           batch_size=self.__batch_size, shuffle=self.__shuffle)

                    if index == 0:
                        print("setting: "+str(len(res['training_error'])))
                        self.set_results(res)
                    else:
                        print("adding: " + str(len(res['training_error'])))
                        self.accumulate_results(res)

                self.get_mean_error()
                self.update_best_results(param_combination)

            elif self.__cv != -1:

                res = self.__model.fit(self.__training_set[0], self.__training_set[1],
                                       validation_data=(
                                           self.__validation_set[0], self.__validation_set[1]),
                                       epochs=self.__epochs,
                                       batch_size=self.__batch_size, shuffle=self.__shuffle)

                self.set_results(res)

                self.update_best_results(param_combination)
            else:
                print(self.__model.optimizer.lr)
                print(self.__model.optimizer.momentum)
                res = self.__model.fit(self.__training_set[0], self.__training_set[1],
                                       epochs=self.__epochs,
                                       batch_size=self.__batch_size, shuffle=self.__shuffle)

                self.results['training_error'] = res['training_error']
                self.results['training_metrics'] = res['training_metrics']

                if self.__best_val is None:
                    self.__best_val = self.results['training_metrics'][-1]
                    self.best_model = self.__model
                    self.__best_params = param_combination
                    self.__best_tr_metric = self.results['training_metrics'][-1]
                    self.__best_tr_loss = self.results['training_error'][-1]
                if self.__current_metric == 'mee':
                    if self.__best_val > self.results['training_metrics'][-1]:
                        self.__best_tr_metric = self.results['training_metrics'][-1]
                        self.best_model = self.__model
                        self.__best_params = param_combination
                        self.__best_tr_loss = self.results['training_error'][-1]
                elif self.__current_metric == 'binary':
                    if self.__best_val < self.results['training_metrics'][-1]:
                        self.__best_val = self.results['training_metrics'][-1]
                        self.__best_tr_metric = self.results['training_metrics'][-1]
                        self.best_model = self.__model
                        self.__best_tr_loss = self.results['training_error'][-1]
                        self.__best_params = param_combination
            for x in self.results['training_error']:
                wandb.log({ "error": x})
        wandb.finish()
        return [self.results, self.__best_params]


    def fit(self, training_data, training_targets, epochs=None, batch_size=None, shuffle=None, cv=3,
            filename='./curr_dataset'):
        wandb.login()
        if cv is not None and cv > 0:
            splitter = KFold()
            self.__training_set, self.__validation_set = splitter.split((training_data, training_targets), cv)
            self.__cv = cv
        elif cv != -1:
            splitter = SimpleHoldout()
            self.__training_set, self.__validation_set = splitter.double_split(training_data, training_targets)
            self.__cv = cv
        else:
            self.__cv = cv
            self.__training_set = [training_data, training_targets]

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

        parallel_split = np.array_split(np.asarray(experiments),  self.__pool_size)
        parallel_args = []
        for x in parallel_split:
            parallel_args.append((x, cv))
        with NestablePool( self.__pool_size) as pool:
            result_pool = pool.map(self.internal_runs, parallel_args)
            pool.close()
            pool.join()

        # print(result_pool)
        # print(len(result_pool))
