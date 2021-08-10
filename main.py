import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from NeuralNetworkCore.Layer import Dense,Dropout
from NeuralNetworkCore.Model import Model
from NeuralNetworkCore.Optimizers import StochasticGradientDescent


def read_cup(int_ts=False):
    """
    Reads the CUP training and test set
    :return: CUP training data, CUP training targets and CUP test data (as numpy ndarray)
    """
    print(os.getcwd())
    # read the datasets
    col_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y']
    directory = "./datasets/cup/"
    int_ts_path = directory + "CUP-INTERNAL-TEST.csv"
    dev_set_path = directory + "CUP-DEV-SET.csv"
    file = "ML-CUP20-TR.csv"

    if int_ts and not (os.path.exists(int_ts_path) and os.path.exists(dev_set_path)):
        df = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(0, 13))
        df = df.sample(frac=1, axis=0)
        int_ts_df = df.iloc[: math.floor(len(df) * 0.1), :]
        dev_set_df = df.iloc[math.floor(len(df) * 0.1):, :]
        int_ts_df.to_csv(path_or_buf=int_ts_path, index=False, float_format='%.6f', header=False)
        dev_set_df.to_csv(path_or_buf=dev_set_path, index=False, float_format='%.6f', header=False)

    if int_ts and os.path.exists(int_ts_path) and os.path.exists(dev_set_path):
        tr_data = pd.read_csv(dev_set_path, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
        tr_targets = pd.read_csv(dev_set_path, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))
        int_ts_data = pd.read_csv(int_ts_path, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
        int_ts_targets = pd.read_csv(int_ts_path, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))
        int_ts_data = int_ts_data.to_numpy(dtype=np.float32)
        int_ts_targets = int_ts_targets.to_numpy(dtype=np.float32)
    else:
        tr_data = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
        tr_targets = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))

    file = "ML-CUP20-TS.csv"
    cup_ts_data = pd.read_csv(directory + file, sep=',', names=col_names[: -2], skiprows=range(7), usecols=range(1, 11))

    tr_data = tr_data.to_numpy(dtype=np.float32)
    tr_targets = tr_targets.to_numpy(dtype=np.float32)
    cup_ts_data = cup_ts_data.to_numpy(dtype=np.float32)

    # shuffle the training datasets once
    indexes = list(range(tr_targets.shape[0]))
    np.random.shuffle(indexes)
    tr_data = tr_data[indexes]
    tr_targets = tr_targets[indexes]

    # detach internal test set if needed
    if int_ts:
        if not (os.path.exists(int_ts_path) and os.path.exists(dev_set_path)):
            tr_data, int_ts_data, tr_targets, int_ts_targets = train_test_split(tr_data, tr_targets, test_size=0.1)

        return tr_data, tr_targets, int_ts_data, int_ts_targets, cup_ts_data

    return tr_data, tr_targets, cup_ts_data


devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)

model = Model("SimpleNet")

model.add(Dense(10, 5))
model.add(Dropout(0.3))
model.add(Dense(5, 1))
optimizer = StochasticGradientDescent(metric='euclidean')
model.compile(optimizer=optimizer)
print(type(devset_x))
print(devset_x.shape)
print(devset_y.shape)
print(devset_y[:, 0].shape)
model.showLayers()
print(model.dense_configuration)
res = model.fit(devset_x, np.reshape(devset_y[:, 0], (-1, 1)), batch_size=10, epochs=20)
plt.plot(res['training_error'])
plt.show()
print(res)
