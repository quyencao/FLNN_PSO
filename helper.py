import numpy as np
from sklearn.preprocessing import MinMaxScaler

def power_data(ori_values):
    p2 = np.power(ori_values, 2)
    p3 = np.power(ori_values, 3)
    p4 = np.power(ori_values, 4)

    scaler = MinMaxScaler(feature_range=(0, 1))

    values = scaler.fit_transform(ori_values)
    p2 = scaler.fit_transform(p2)
    p3 = scaler.fit_transform(p3)
    p4 = scaler.fit_transform(p4)

    return values, p2, p3, p4


def chebyshev_data(ori_values):
    c2x2 = 2 * np.power(ori_values, 2) - 1
    c4x3 = 4 * np.power(ori_values, 3) - 3 * ori_values
    c8x4 = 8 * np.power(ori_values, 4) - 8 * np.power(ori_values, 2) + 1

    scaler = MinMaxScaler(feature_range=(0, 1))

    values = scaler.fit_transform(ori_values)
    c2x2 = scaler.fit_transform(c2x2)
    c4x3 = scaler.fit_transform(c4x3)
    c8x4 = scaler.fit_transform(c8x4)

    return values, c2x2, c4x3, c8x4


def legendre_data(ori_values):
    c2x2 = (3 * np.power(ori_values, 2) - 1) / 2
    c4x3 = (5 * np.power(ori_values, 3) - 3 * ori_values) / 2
    c8x4 = (35 * np.power(ori_values, 4) - 30 * np.power(ori_values, 2) + 3) / 8

    scaler = MinMaxScaler(feature_range=(0, 1))

    values = scaler.fit_transform(ori_values)
    c2x2 = scaler.fit_transform(c2x2)
    c4x3 = scaler.fit_transform(c4x3)
    c8x4 = scaler.fit_transform(c8x4)

    return values, c2x2, c4x3, c8x4


def laguerre_data(ori_values):
    c2x2 = -ori_values + 1
    c4x3 = np.power(ori_values, 2) / 2 - ori_values + 1
    c8x4 = ((5 - ori_values) * c4x3 - 2 * c2x2) / 3

    scaler = MinMaxScaler(feature_range=(0, 1))

    values = scaler.fit_transform(ori_values)
    c2x2 = scaler.fit_transform(c2x2)
    c4x3 = scaler.fit_transform(c4x3)
    c8x4 = scaler.fit_transform(c8x4)

    return values, c2x2, c4x3, c8x4