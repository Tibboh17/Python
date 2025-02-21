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

# RNN class
class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.w_x = np.random.randn(self.hidden_dim, self.input_dim) * 0.01
        self.w_h = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
        self.w_y = np.random.randn(self.output_dim, self.hidden_dim) * 0.01
        
        self.b_h = np.zeros((self.hidden_dim, 1))
        self.b_y = np.zeros((self.output_dim, 1))

    def step(self, x_t, h_prev):
        h_t = np.tanh(self.w_x @ x_t + self.w_h @ h_prev + self.b_h)
        y_t = self.w_y @ h_t + self.b_y
        return y_t, h_t
    
    def forward(self, x):
        self.x = x
        self.T = x.shape[0]
        h_prev = np.zeros((self.hidden_dim, 1))

        y_pred = []
        h_states = []

        for t in range(self.T):
            x_t = self.x[t].reshape(-1, 1)
            y_t, h_t = self.step(x_t, h_prev)
            h_prev = h_t

            y_pred.append(y_t)
            h_states.append(h_t)

        return np.array(y_pred).reshape(-1, 1), h_states
    
    def backward(self, y_true, y_pred, h_states, learning_rate, clip_value=1.0):
        dw_x = np.zeros_like(self.w_x)
        dw_h = np.zeros_like(self.w_h)
        dw_y = np.zeros_like(self.w_y)
        
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dy = mse_derivative(y_true, y_pred)

        dh_next = np.zeros_like(h_states[0])

        for t in reversed(range(self.T)):
            x_t = self.x[t].reshape(1, -1)
            
            dy_t = dy[t].reshape(-1, 1)
            dw_y += dy_t @ h_states[t].T
            db_y += dy_t

            dh = self.w_y.T @ dy_t + dh_next
            dh_raw = (1 - h_states[t] ** 2) * dh

            dw_x += dh_raw @ x_t
            if t > 0:
                dw_h += dh_raw @ h_states[t - 1].T
            else:
                dw_h += dh_raw @ np.zeros_like(h_states[t]).T
            db_h += dh_raw

            dh_next = self.w_h.T @ dh_raw
        
        for gradient in [dw_x, dw_h, dw_y, db_h, db_y]:
            np.clip(gradient, -clip_value, clip_value, out=gradient)

        self.w_x -= learning_rate * dw_x
        self.w_h -= learning_rate * dw_h
        self.w_y -= learning_rate * dw_y

        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y