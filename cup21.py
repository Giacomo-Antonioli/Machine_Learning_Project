from multiprocessing import freeze_support
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

from NeuralNetworkCore.Layer import Dense, Dropout
from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import StochasticGradientDescent
from NeuralNetworkCore.Validation.Model_selection import GridSearch
# devi scrivere qua per il cup
from NeuralNetworkCore.Utils.LoadCSVData import LoadCSVData

training_set_file = "ML-CUP21-TR.csv" 
test_set_file = "ML-CUP21-TS.csv"
cupPath = "./datasets/cup/"

opt = ['sgd']
mom = [0.6]
lr = [0.001]

metrics = ['mee']
loss = ['squared']
beta1 = [0.9]
beta2 = [0.999]
epsilon = [1e-8]
rho = [0.95, 0.9]

n_layer = 3

n_unit_1 = [30]
n_unit_2 = [10]
n_unit_3 = [1]
ac_fun_1 = ['sigmoid']
ac_fun_2 = ['sigmoid']
ac_fun_3 = ['linear']
weight_init = ['glorot_uniform']
bias_init = ['glorot_uniform']
# drop_percent = [0.1,0.2,0.3]
epochs = 300

cross_validation = 5
batch_size = [32]
reg = ['l1']
reg_param = [1e-5]

cup_dataset, cup_lables= LoadCSVData.loadCSV(path = "datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_names=None, column_for_label=10, drop_rows=[0,1,2,3,4,5,6], drop_cols = 11)
cup_dataset1, cup_lables1= LoadCSVData.loadCSV(path = "datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_names=None, column_for_label=11, drop_rows=[0,1,2,3,4,5,6], drop_cols = 10)

model = Model("cup2_correct")

''' model.create_net(num_layer = n_layer, drop_frequency=1, num_unit=[10], act_func=['linear'], weight_init= ['glorot_normal'],
                     bias_init=['glorot_normal'], drop_percentage=[0.3]) '''
model.set_input_shape(10)

model.add(Dense(30, regularizer=('l1', 0.7), activation_function='sigmoid'))
model.add(Dense(10, regularizer=('l1', 0.7), activation_function='sigmoid'))
model.add(Dense(1, activation_function='linear'))

optimizer = StochasticGradientDescent()
model.compile(optimizer=optimizer, metrics='mee', loss='squared')

gridsearch_1 = GridSearch(model,
                            {'opt': opt,
                              'mom': mom,
                              'lr': np.true_divide(lr,batch_size),
                              'metrics': metrics,
                              'loss': loss,
                              'b1': beta1,
                              'b2': beta2,
                              'epsilon': epsilon,
                              'rho': rho,
                              'weightinit_1': weight_init,
                              'weightinit_2': weight_init,
                              "weightinit_3": weight_init,
                              'biasinit_1': bias_init,
                              'biasinit_2': bias_init,
                              'biasinit_3': bias_init,
                              'actfun_1': ac_fun_1,
                              'actfun_2': ac_fun_2,
                              'actfun_3': ac_fun_3,
                              'units_1': n_unit_1,
                              'units_2': n_unit_2,
                              'units_3': n_unit_3,
                              'batchsize': batch_size,
                              'reg_1': reg,
                              'regparam_1': reg_param,
                              'reg_2': reg,
                              'regparam_2': reg_param
                            }
                            )

if __name__ == '__main__':

    gridsearch_1.fit(cup_dataset, cup_lables, epochs=epochs, batch_size=batch_size, shuffle=True, cv=cross_validation)
