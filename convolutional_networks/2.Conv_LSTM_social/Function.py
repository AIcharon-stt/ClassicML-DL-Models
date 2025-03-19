# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/9 13:08
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import pandas as pd
import matplotlib.pyplot as plt


# 数据预处理函数
def preprocess_data(time_series_data, social_data, sequence_length):
    # 归一化时间序列数据
    scaler_time = MinMaxScaler()
    time_series_normalized = scaler_time.fit_transform(time_series_data)

    # 处理社会因素数据
    social_data = social_data.set_index("ID")
    scaler_social = MinMaxScaler()
    social_data_normalized = scaler_social.fit_transform(social_data)

    # 创建区域ID到社会因素的映射
    social_factors_map = {idx: factors for idx, factors in zip(social_data.index, social_data_normalized)}

    x, y, social_features = [], [], []
    for i in range(sequence_length, len(time_series_normalized) - 1):
        _x = time_series_normalized[i - sequence_length:i]
        _y = time_series_normalized[i]
        region_id = int(time_series_data.iloc[i, 0])  # 假设第一列是区域ID
        _social = social_factors_map.get(region_id, np.zeros(social_data.shape[1]))
        x.append(_x)
        y.append(_y)
        social_features.append(_social)

    return np.array(x), np.array(y), np.array(social_features), scaler_time, scaler_social


# 损失计算函数
def calculate_loss(outputs, y_train):
    criterion = nn.MSELoss()
    return criterion(outputs, y_train)


# 模型训练函数
def train_model(model, optimizer, num_epochs, x_train, y_train, social_train):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()
        outputs = model(x_train, social_train)
        loss = calculate_loss(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Duration: {end_time - start_time}s")
    return train_losses


# 模型评估函数
def evaluate_model(model, x_test, y_test, social_test, scaler_time, model_type):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test, social_test)
        predictions = scaler_time.inverse_transform(predictions.cpu().numpy())
        y_test_original = scaler_time.inverse_transform(y_test.cpu().numpy())

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

        # 计算每一列的MPE并输出到CSV文件
        col_mpe = []
        for i in range(y_test_original.shape[1]):
            col_mask = y_test_original[:, i] > 0.1
            if np.any(col_mask):
                col_mpe.append(np.mean(
                    (y_test_original[:, i][col_mask] - predictions[:, i][col_mask]) / y_test_original[:, i][
                        col_mask]) * 100)
            else:
                col_mpe.append(0)

        pd.DataFrame(col_mpe, index=[f'{i}' for i in range(len(col_mpe))]).to_csv(
            f'results/mpe_values_average_social2_{model_type}.csv',
            header=['MPE'])

        # 计算每个元素的MPE并输出到CSV文件
        mpe_by_element = np.where(y_test_original < 0.1, 0,
                                  (y_test_original - predictions) / (y_test_original + 1e-8) * 100)

        # Output MPE by element to a CSV file
        df_mpe_by_element = pd.DataFrame(mpe_by_element)
        df_mpe_by_element.to_csv(f'results/mpe_values_social2_{model_type}.csv', index=False)

        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'MPE': mpe}, predictions, y_test_original


# 绘制预测和真实值拟合曲线函数
def plot_predictions(predictions, y_test_original, title):
    plt.figure(figsize=(12, 6))
    for i in range(predictions.shape[1]):
        plt.plot(predictions[:, i], label=f'Prediction {i + 1}')
        plt.plot(y_test_original[:, i], label=f'Actual {i + 1}', linestyle='--')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'result/{title}.png')
    plt.close()
