import numpy as np
import pandas as pd

from NeuralNetworkCore.Layer import Dense, Dropout
from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import RMSProp
from NeuralNetworkCore.Validation.Model_selection import GridSearch

col_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y']
directory = "./datasets/cup/"
int_ts_path = directory + "CUP-INTERNAL-TEST.csv"
dev_set_path = directory + "CUP-DEV-SET.csv"
file = "ML-CUP20-TR.csv"

tr_data = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
tr_targets = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))

file = "ML-CUP20-TS.csv"
cup_ts_data = pd.read_csv(directory + file, sep=',', names=col_names[: -2], skiprows=range(7), usecols=range(1, 11))

devset_x = tr_data.to_numpy(dtype=np.float32)
devset_y = tr_targets.to_numpy(dtype=np.float32)
cup_ts_data = cup_ts_data.to_numpy(dtype=np.float32)
data_trainx = devset_x[:int(len(devset_x) * 0.9)]
data_internal_testx = devset_x[-int(len(devset_x) * 0.1):]
data_trainy_1 = np.reshape(devset_y[:int(len(devset_y) * 0.9), 0], (-1, 1))
data_internal_testy_1 = np.reshape(devset_y[-int(len(devset_y) * 0.1):, 0], (-1, 1))
data_trainy_2 = np.reshape(devset_y[:int(len(devset_y) * 0.9), 1], (-1, 1))
data_internal_testy_2 = np.reshape(devset_y[-int(len(devset_y) * 0.1):, 1], (-1, 1))

model = Model("CUP1")
model.set_input_shape(10)
model.add(Dense(5, regularizer=('l1', 0.9), activation_function='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(3, regularizer=('l2', 0.001), activation_function='relu'))
model.add(Dense(1))
optimizer = RMSProp()
model.compile(optimizer=optimizer, metrics='accuracy', loss='squared', early_stopping=False, mode='absolute_growth')
model.showLayers()

gridsearch_1 = GridSearch(model,
                          {'dense_1': [5, 10, 15], 'regparam_1': [0.001], 'lr': [0.1, 0.01], 'drop_1': [0.1, 0.3],
                           'mom': [0.1, 0.5], 'reg_1': ['l1', 'l2'], 'opt': ['sgd', 'rmsprop']}
                          )
gridsearch_1.fit(data_trainx, data_trainy_1, epochs=[50], batch_size=20, shuffle=False, cv=3, filename='cup1')

best_1 = gridsearch_1.best_model
int_test_1 = best_1.evaluate(data_internal_testx, data_internal_testy_1)
best1_predicted = best_1.predict(cup_ts_data)

model = Model("CUP2")
model.set_input_shape(10)
model.add(Dense(5, regularizer=('l1', 0.9), activation_function='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(1))
optimizer = RMSProp()
model.compile(optimizer=optimizer, metrics='accuracy', loss='squared', early_stopping=False, mode='absolute_growth')
model.showLayers()
data_trainx = devset_x
data_trainy_1 = np.reshape(devset_y[:, 0], (-1, 1))
data_trainy_2 = np.reshape(devset_y[:, 1], (-1, 1))

gridsearch_2 = GridSearch(model,
                          {'dense_1': [10], 'regparam_1': [0.001], 'lr': [0.01], 'drop_1': [0.1],
                           'mom': [0.1], 'reg_1': ['l2'], 'opt': ['rmsprop']})
gridsearch_2.fit(data_trainx, data_trainy_2, epochs=[50], batch_size=10, shuffle=False, cv=3, filename='cup2')

best_2 = gridsearch_2.best_model
int_test_2 = best_2.evaluate(data_internal_testx, data_internal_testy_2)
best2_predicted = best_2.predict(cup_ts_data)

print("#######################################")
print("Best TR metric 1")
print(gridsearch_1.best_tr_metric)
print("Best Val metric 1")
print(gridsearch_1.best_val_metric)
print("Best Int metric 1")
print(int_test_1[1])
# print("#######################################")
print("Best TR metric 2")
print(gridsearch_2.best_tr_metric)
print("Best Val metric 1")
print(gridsearch_2.best_val_metric)
print("Best Int metric 2")
print(int_test_2[1])

# f = open('./results/Test_set_results.txt', 'w+')
# f.writelines('# Giacomo Antonioli')
# f.write('\n')
# f.writelines('# DonzTeam')
# f.write('\n')
# f.writelines('# ML-CUP20')
# f.write('\n')
# f.writelines('# 22/08/2021')
# for index,x in enumerate(best1_predicted):
#     f.writelines(str(index+1)+'\t'+str(x)+'\t'+str(best2_predicted[index])+'\n')
# f.close()
