#  Copyright (c) 2021.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from multiprocessing import freeze_support
from pickle import ADDITEMS
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

from NeuralNetworkCore.Layer import Dense
from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import StochasticGradientDescent, Adam
from NeuralNetworkCore.Validation.Model_selection import GridSearch

from NeuralNetworkCore.Utils.LoadCSVData import LoadCSVData
import wandb


columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']

#hyper-parameters
opt = ['adam']
mom = [0.7]
lr = [ 0.00001]
metrics = ['binary']
loss = ['squared']
beta1 = [0.9]
beta2 = [0.999]
epsilon = [1e-9, 1e-8]
rho = [0.95, 0.9]

n_layer = 2
n_unit_1 = [10]
n_unit_2 = [1]
ac_fun_1 = ['tanh']
ac_fun_2 = ['sigmoid']
weight_init = ['uniform']
bias_init = ['uniform']
drop_percent = [0.3,0.5]
epochs = 400
cross_validation = 3
batch_size = [ 1]
reg = ['l1']
reg_param = [0.0003]

for monk in ['monks-3']:#, 'monks-2', 'monks-3'
    monk_train = monk + '.train'
    monk_test = monk + '.test'
    
    if __name__ == '__main__':
        print("---------------------------MODEL---------------------------------")
        model.showLayers()
        print("-----------------------------------------------------------------")
        monk_dataset, monk_labels= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_train), separator=' ', column_names=columns, column_for_label=0, returnFit=True)   
        monk_dataset_test, monk_labels_test= LoadCSVData.loadCSV(path = "./datasets/monks/", file_name = str(monk_test), separator=' ', column_names=columns, column_for_label=0, returnFit=True)

        model = Model(monk)
        model.set_input_shape(17)
        
        model.add(Dense(15, activation_function='tanh',weight_initializer='he_uniform',regularizer=('l1',0.01)))
        model.add(Dense(1, activation_function='sigmoid'))
        optimizer = Adam()
        model.compile(optimizer=optimizer, metrics='binary', loss='squared')
        res=model.fit(monk_dataset,monk_labels,validation_data=(monk_dataset_test, monk_labels_test), epochs=400, batch_size=30, shuffle=True)
        wandb.login()
        wandb.init(
            #Set entity to specify your username or team name
            entity="ml_project",
            #Set the project where this run will be logged
            
            project="test._._" + model.name+"_with_reg",
            group="experiment_" + model.name,
    	    #Track hyperparameters and run metadata
            reinit=True)
        print(res["training_error"][-1])
        print(res["training_metrics"][-1])
        print(res["validation_error"][-1])
        print(res["validation_metrics"][-1])
        for index,x in enumerate(res["training_error"]):
            wandb.log({ #'Epoch': epoch,
                        "Train Loss": res['training_error'][index],
                        "Train Acc ":res['training_metrics'][index],
                        "Valid Loss": res['validation_error'][index],
                        "Valid Acc ":res['validation_metrics'][index]
            })
        wandb.finish()  
        print("Done")
