#  Copyright (c) 2021.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from multiprocessing import freeze_support
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

from NeuralNetworkCore.Layer import Dense
from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import StochasticGradientDescent
from NeuralNetworkCore.Validation.Model_selection import GridSearch

from NeuralNetworkCore.Utils.LoadCSVData import LoadCSVData



columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']

#hyper-parameters
opt = ['adam', 'rmsprop', 'sgd']
mom = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
metrics = ['mee']
loss = ['squared']
beta1 = [0.95, 0.9]
beta2 = [0.999, 0.995]
epsilon = [1e-9, 1e-8]
rho = [0.95, 0.9]

n_layer = 2
n_unit_1 = [20]
n_unit_2 = [1]
ac_fun_1 = ['tanh']
ac_fun_2 = ['sigmoid']
weight_init = ['glorot_normal']
bias_init = ['glorot_normal']
drop_percent = [0.3]
epochs = 100
cross_validation = 5
batch_size = [1, 5]
reg = ['l1']
reg_param = [0.0003]

for monk in ['monks-1']:#, 'monks-2', 'monks-3'
    monk_train = monk + '.train'
    monk_test = monk + '.test'


    print(os.getcwd())
    monk_dataset, monk_labels= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_train), separator=' ', column_names=columns, column_for_label='class', returnFit=True)   
    monk_dataset_test, monk_labels_test= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_test), separator=' ', column_names=columns, column_for_label='class', returnFit=True)

    model = Model(monk)
    model.set_input_shape(17)
    model.create_net(num_layer = n_layer, drop_frequency=1, num_unit=[30], act_func=['tanh', 'sigmoid'], weight_init= ['glorot_normal'],
                     bias_init=['glorot_normal'], drop_percentage=[0.3, 0.4], drop_seed=[10])
    
    #model.add(Dense(30, activation_function='tanh'))
    #model.add(Dense(40, activation_function='sigmoid'))
    optimizer = StochasticGradientDescent()
    model.compile(optimizer=optimizer, metrics='binary', loss='squared')
    
    gridsearch_1 = GridSearch(model,
                              {'opt': opt,
                               'mom': mom,
                               'lr': lr,
                               'metrics': metrics,
                               'loss': loss,
                               'b1': beta1,
                               'b2': beta2,
                               'epsilon': epsilon,
                               'rho': rho,
                               'weightinit_1': weight_init,
                               'weightinit_2': weight_init,
                               'biasinit_1': bias_init,
                               'biasinit_2': bias_init,
                               'actfun_1': ac_fun_1,
                               'actfun_2': ac_fun_2,
                               'units_1': n_unit_1,
                               'units_2': n_unit_2,
                               'batchsize': batch_size,
                               'reg_1': reg,
                               'regparam_1': reg_param,
                               'reg_2': reg,
                               'regparam_2': reg_param
                               }
                            )
    
    if __name__ == '__main__':
        gridsearch_1.fit(monk_dataset, monk_labels, epochs=epochs, batch_size=batch_size, shuffle=True, cv=cross_validation)
        print("Done")
        ''' best_1=gridsearch_1.best_model
        # int_test_1=best_1.evaluate(monk_dataset_test,monk_labels_test)
        print("#######################################")
        print(monk)
        print("Best TR metric")
        print(gridsearch_1.best_tr_metric)
        print("Best TR loss")
        print(gridsearch_1.best_tr_loss)
        print("Best Int metric")
        print(int_test_1[1])
        print("Best Int Loss")
        print(int_test_1[0])
        print("Best params")
        print(gridsearch_1.best_params)
        input() '''

