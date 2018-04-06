import pso
import helper
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

random.seed(42)

colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space']
df = pd.read_csv('Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[0])
df.dropna(inplace=True)

ori_values = df.values

values, p2, p3, p4 = helper.power_data(ori_values)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(ori_values)

window = 4
n = values.shape[0]

X = []
y = []
for i in range(n - window):
    row = []

    for j in range(window):
        row.append(values[i + j, 0])

    for j in range(window):
        row.append(p2[i + j, 0])

    for j in range(window):
        row.append(p3[i + j, 0])

    for j in range(window):
        row.append(p4[i + j, 0])

    X.append(row)
    y.append(values[i + window, 0])
X = np.array(X)
y = np.array(y).reshape((-1, 1))

train_size = int(0.6 * X.shape[0])
valid_size = int(0.2 * X.shape[0])
X_train_tr, y_train, X_valid, y_valid, X_test_tr, y_test = X[:train_size, :], y[:train_size, :], X[train_size:train_size+valid_size, :], y[train_size:train_size+valid_size, :], X[train_size+valid_size:, :], ori_values[train_size+valid_size+window:, :]

X_train_tr = X_train_tr.T
X_test_tr = X_test_tr.T
X_valid = X_valid.T
y_train = y_train.T
y_test = y_test.T
y_valid = y_valid.T


if __name__ == '__main__':
    p = pso.PSO(X_train_tr, y_train, X_valid, y_valid)
    best_global_solutions = p.train(200)

    maes = []

    for s in best_global_solutions:
        predicts = s.predict(X_test_tr)

        unnorm_predicts = scaler.inverse_transform(predicts)

        mae = mean_absolute_error(unnorm_predicts, y_test)

        maes.append(mae)

    print("MAE: %.5f" % (min(maes)))