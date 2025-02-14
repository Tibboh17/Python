import numpy as np
from activate import *
from loss import *

# MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w_ih = np.random.randn(input_dim, hidden_dim) * 0.01
        self.w_ho = np.random.randn(hidden_dim, output_dim) * 0.01
        
        self.b_h = np.zeros((1, hidden_dim))
        self.b_o = np.zeros((1, output_dim))

    def forward(self, x):
        self.x = x

        z = np.dot(x, self.w_ih) + self.b_h
        a = relu(z)
        
        y_pred = np.dot(a, self.w_ho) + self.b_o
        cache = (z, a)

        return y_pred, cache
    
    def backward(self, y_true, y_pred, cache, learning_rate):
        z, a = cache
        dy = mse_derivative(y_true, y_pred)

        dw_ho = np.dot(a.T, dy)
        db_o = np.sum(dy, axis=0, keepdims=True)

        da = np.dot(dy, self.w_ho.T)
        dz = da * relu_derivative(z)

        dw_ih = np.dot(self.x.T, dz)
        db_h = np.sum(dz, axis=0, keepdims=True)

        self.w_ih -= learning_rate * dw_ih
        self.w_ho -= learning_rate * dw_ho
        self.b_h -= learning_rate * db_h
        self.b_o -= learning_rate * db_o