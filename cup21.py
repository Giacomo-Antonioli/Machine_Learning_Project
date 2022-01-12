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
#vediamo la tua magia
#ok
#i label dovrebbero essere 2 vettori
#le ultime 2 colonne

cup_dataset, cup_lables= LoadCSVData.loadCSV(path = "datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_names=None, column_for_label=10, drop_rows=[0,1,2,3,4,5,6], drop_cols = [11])
cup_dataset1, cup_lables1= LoadCSVData.loadCSV(path = "datasets/cup/", file_name = 'ML-CUP21-TR.csv', separator=',', column_names=None, column_for_label=11, drop_rows=[0,1,2,3,4,5,6], drop_cols = [10])

''' print(cup_dataset)
print("--------")
print(cup_lables) '''

model = Model("cup1")

model.set_input_shape(10)
model.add(Dense(4, regularizer=('l1', 0.009), activation_function='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation_function='linear'))
optimizer = StochasticGradientDescent()
model.compile(optimizer=optimizer, metrics='mee', loss='squared')

gridsearch_1 = GridSearch(model,
                            { 'lr': [0.1], 'drop_1': [0.1, 0.3],
                           'mom': [0.1, 0.5], 'opt': ['sgd', 'rmsprop']}
                            )

if __name__ == '__main__':
    gridsearch_1.fit(cup_dataset, cup_lables, epochs=100, batch_size=32, shuffle=False, cv=3)