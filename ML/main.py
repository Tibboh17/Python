import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import *

def standarize(data, train_mean, train_std):
    return (data - train_mean) / train_std

def min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def train(model, x, y, epochs, learning_rate):
        loss_list = []

        for epoch in range(epochs):
            epoch_loss = 0

            y_pred, cache = model.forward(x)
            model.backward(y, y_pred, cache, learning_rate)

            loss = mse(y, y_pred)
            epoch_loss += loss
            loss_list.append(epoch_loss / x.shape[0])
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss}')

        return loss_list

def metrics(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)

        r_square = 1 - (ss_residual / ss_total)
        rmse = np.sqrt(mse(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs(y_true - y_pred) / y_true) * 100

        print(f'R-Square: {r_square:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'MAPE: {mape:.4f}')


file_path = './Datasets/조업편차분석.csv'
data = pd.read_csv(file_path)

data['Origin'] = data.index
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(len(shuffled_data) * 0.8)
train_data, test_data = shuffled_data.iloc[:train_size], shuffled_data.iloc[train_size:]

x_train, y_train = train_data.drop(columns=['No.', 'A1', 'Origin'], axis=1), train_data['A1']

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
y_train_mean = np.mean(y_train, axis=0)
y_train_std = np.std(y_train, axis=0)

x_train = standarize(x_train, x_train_mean, x_train_std).to_numpy()
y_train = standarize(y_train, y_train_mean, y_train_std).to_numpy().reshape(-1, 1)

test_data_sorted = test_data.sort_values(by='Origin').reset_index(drop=True)

x_test_sorted, y_test_sorted = test_data_sorted.drop(columns=['No.', 'A1', 'Origin'], axis=1), test_data_sorted['A1']
x_test_sorted = standarize(x_test_sorted, x_train_mean, x_train_std).to_numpy()
y_test_sorted = standarize(y_test_sorted, y_train_mean, y_train_std).to_numpy().reshape(-1, 1)


input_dim = x_train.shape[1]
hidden_dim = 64
output_dim = y_train.shape[1]

# MLP
mlp_model = MLP(
    input_dim=input_dim, 
    hidden_dim=hidden_dim,
    output_dim=output_dim
)

# RNN
rnn_model = RNN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)

loss_list = train(rnn_model, x_train, y_train, epochs=1000, learning_rate=0.01)
y_pred, _ = rnn_model.forward(x_test_sorted)

plt.figure(figsize=(10, 6))
plt.plot(loss_list)

plt.figure(figsize=(40, 6))
plt.plot(y_test_sorted, marker='x')
plt.plot(y_pred, marker='o')
plt.show()