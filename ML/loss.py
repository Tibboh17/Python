import numpy as np

def mse(true, pred):
    return np.mean((pred - true) ** 2)

def mse_derivative(true, pred):
    return 2 * (pred - true) / true.size