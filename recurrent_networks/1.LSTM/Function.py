# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/8 15:36
# ------------------------------------------------------------------------------

# 模型训练函数
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


# 数据预处理函数
def preprocess_demand_data(data_column, sequence_length):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_column.values.reshape(-1, 1))

    x, y = [], []
    for i in range(len(data_normalized) - sequence_length - 1):
        _x = data_normalized[i:(i + sequence_length)]
        _y = data_normalized[i + sequence_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y), scaler


# 模型训练函数
def train_model(model, optimizer, num_epochs, x_train, y_train):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()
        outputs = model(x_train)
        optimizer.zero_grad()
        loss = calculate_loss(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Duration: {end_time - start_time}s")
    return train_losses


# 损失计算函数
def calculate_loss(outputs, y_train):
    criterion = nn.MSELoss()
    return criterion(outputs, y_train)


# 模型评估函数
def evaluate_model(model, x_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        predictions = scaler.inverse_transform(predictions.cpu().numpy())
        y_test_original = scaler.inverse_transform(y_test.cpu().numpy())

        mse = mean_squared_error(y_test_original, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, predictions)
        r2 = r2_score(y_test_original, predictions)

        mask = y_test_original > 0.1
        if np.any(mask):
            mape = np.mean(np.abs((y_test_original[mask] - predictions[mask]) / y_test_original[mask])) * 100
            mpe = np.mean((y_test_original[mask] - predictions[mask]) / y_test_original[mask]) * 100

        else:
            mape = 0
            mpe = 0

        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'MPE': mpe,'MPE_values': 100*(y_test_original[mask] - predictions[mask]) / y_test_original[mask]}, predictions, y_test_original, mpe


# 绘制预测和真实值拟合曲线函数
def plot_predictions(predictions, y_test_original, column_index):
    plt.figure()
    plt.plot(predictions, label='Predictions')
    plt.plot(y_test_original, label='Actual Data')
    plt.title(f'Predictions vs Actual Data for Column {column_index}')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'result/predictions_column_{column_index}.png')
    plt.close()
