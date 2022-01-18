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
''' mom = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9]
lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9]
metrics = ['mee']
loss = ['squared']
beta1 = [0.95, 0.9, 0.85, 0.8]
beta2 = [0.999, 0.995, 0.99, 0.95]
epsilon = [1e-9, 1e-8]
rho = [0.95, 0.9, 0.85, 0.8] '''
mom = [0.1]
lr = [0.1]
metrics = ['mee']
loss = ['squared']

cup_dataset, cup_lables= LoadCSVData.loadCSV(path = "datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_names=None, column_for_label=10, drop_rows=[0,1,2,3,4,5,6], drop_cols = [11])
cup_dataset1, cup_lables1= LoadCSVData.loadCSV(path = "datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_names=None, column_for_label=11, drop_rows=[0,1,2,3,4,5,6], drop_cols = [10])

''' print(cup_dataset)
print("--------")
print(cup_lables) '''

model = Model("cup1")

model.create_net(num_layer = 4, drop_frequency=1, num_unit=[4], act_func=['relu'], weight_init= ['glorot_normal'],
                     bias_init=['glorot_normal'], drop_percentage=[0.3], drop_seed=[10])
model.set_input_shape(10)

''' model.set_input_shape(10)
model.add(Dense(20, regularizer=('l1', 0.0003), activation_function='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation_function='linear')) '''
optimizer = StochasticGradientDescent()
model.compile(optimizer=optimizer, metrics='mee', loss='squared')

''' gridsearch_1 = GridSearch(model,
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
                            ) '''
gridsearch_1 = GridSearch(model,
                            {'opt': opt,
                            'mom': mom,
                            'lr': lr,
                            'metrics': metrics,
                            'loss': loss,
                            }
                            )

if __name__ == '__main__':
    gridsearch_1.fit(cup_dataset, cup_lables, epochs=600, batch_size=32, shuffle=True, cv=5)