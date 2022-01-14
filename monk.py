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

#parameters
opt = ['adam', 'rmsprop', 'sgd']
mom = [0.1, 0.2]
lr = [0.1]
metrics = ['binary']
loss = ['squared']
beta1 = [0.9, 0.8]
beta2 = [0.999, 0.888]
epsilon = [1e-9]
rho = [0.9, 0.8]

for monk in ['monks-1']:#, 'monks-2', 'monks-3'
    monk_train = monk + '.train'
    monk_test = monk + '.test'


    print(os.getcwd())
    monk_dataset, monk_labels= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_train), separator=' ', column_names=columns, column_for_label='class', returnFit=True)   
    monk_dataset_test, monk_labels_test= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_test), separator=' ', column_names=columns, column_for_label='class', returnFit=True)

    model = Model(monk)
    model.set_input_shape(17)
    model.create_net(num_layer = 4, drop_frequency=1, num_unit=[4], act_func=['tanh', 'sigmoid'], weight_init= ['glorot_uniform', 'glorot_normal'],
                     bias_init=['glorot_normal', 'glorot_uniform'], drop_percentage=[0.3, 0.4], drop_seed=[10])
    
    #model.add(Dense(4, activation_function='tanh'))
    #model.add(Dense(1, activation_function='sigmoid'))
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
                               'e': epsilon,
                               'rho': rho
                               }
                              )
    
    if __name__ == '__main__':
        gridsearch_1.fit(monk_dataset, monk_labels, epochs=100, batch_size=20, shuffle=True, cv=3)
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

