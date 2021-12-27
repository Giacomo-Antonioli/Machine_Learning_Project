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
from sklearn.preprocessing import OneHotEncoder

from NeuralNetworkCore.Layer import Dense
from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import StochasticGradientDescent
from NeuralNetworkCore.Validation.Model_selection import GridSearch

from NeuralNetworkCore.Utils.LoadCSVData import LoadCSVData



columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
for monk in ['monks-1', 'monks-2', 'monks-3']:
    monk_train = monk + '.train'
    monk_test = monk + '.test'


    #print(os.getcwd())
    monk_dataset, monk_labels= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_train), separator=' ', column_names=columns, column_for_label=0, returnFit=True)   
    monk_dataset_test, monk_labels_test= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_test), separator=' ', column_names=columns, column_for_label='class', returnFit=True)

    print("-here-")
    print(monk_labels)
    print(monk_dataset)

    model = Model(monk)
    model.set_input_shape(17)
    model.add(Dense(4, activation_function='tanh'))
    model.add(Dense(1, activation_function='sigmoid'))
    optimizer = StochasticGradientDescent()
    model.compile(optimizer=optimizer, metrics='binary', loss='squared')

    gridsearch_1 = GridSearch(model,
                              {'opt': ['sgd', 'rmsprop'], 'mom': [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
                               'lr': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'metrics': ['binary'], 'loss': ['squared']}
                              )
    if __name__ == '__main__':
        gridsearch_1.fit(monk_dataset, monk_labels, epochs=10, batch_size=10, shuffle=False, cv=3, filename=monk)
    # best_1=gridsearch_1.best_model
    # int_test_1=best_1.evaluate(monk_dataset_test,monk_labels_test)
    # print("#######################################")
    # print(monk)
    # print("Best TR metric")
    # print(gridsearch_1.best_tr_metric)
    # print("Best TR loss")
    # print(gridsearch_1.best_tr_loss)
    # print("Best Int metric")
    # print(int_test_1[1])
    # print("Best Int Loss")
    # print(int_test_1[0])
    # print("Best params")
    # print(gridsearch_1.best_params)
    # input()

