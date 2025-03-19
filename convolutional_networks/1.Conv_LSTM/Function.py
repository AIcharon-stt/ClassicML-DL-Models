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
def preprocess_data(combined_data, sequence_length):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(combined_data)

    x, y = [], []
    for i in range(len(data_normalized) - sequence_length - 1):
        _x = data_normalized[i:(i + sequence_length)]
        _y = data_normalized[i + sequence_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y), scaler


# 损失计算函数
def calculate_loss(outputs, y_train):
    criterion = nn.MSELoss()
    return criterion(outputs, y_train)


# 模型训练函数
def train_model(model, optimizer, num_epochs, x_train, y_train):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()
        outputs = model(x_train)
        loss = calculate_loss(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Duration: {end_time - start_time}s")
    return train_losses


def evaluate_model(model, x_test, y_test, scaler_demand, model_type):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        # 使用单独的需求数据scaler来逆转预测值的归一化
        predictions = scaler_demand.inverse_transform(predictions.cpu().numpy())

        y_test_original = scaler_demand.inverse_transform(y_test.cpu().numpy())

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

        pd.DataFrame(col_mpe, index=[f'{i}' for i in range(len(col_mpe))]).to_csv(f'results/mpe_values_average_{model_type}.csv',
                                                                                         header=['MPE'])

        # 计算每个元素的MPE并输出到CSV文件
        mpe_by_element = np.where(y_test_original < 0.1, 0,
                                  (y_test_original - predictions) / (y_test_original + 1e-8) * 100)

        # Output MPE by element to a CSV file
        df_mpe_by_element = pd.DataFrame(mpe_by_element)
        df_mpe_by_element.to_csv(f'results/mpe_values_{model_type}.csv', index=False)

        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'MPE': mpe}, predictions, y_test_original

def evaluate_model1(model, x_test, y_test, scaler_demand, model_type):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        # 使用单独的需求数据scaler来逆转预测值的归一化
        predictions = scaler_demand.inverse_transform(predictions.cpu().numpy())

        y_test_original = scaler_demand.inverse_transform(y_test.cpu().numpy())

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

        col_mpe_series = pd.Series(col_mpe, index=[f'{i}' for i in range(len(col_mpe))])
        col_mpe_series.to_csv(f'results/mpe_values_average_{model_type}.csv', header=['MPE'])

        # 设置可达性ID
        low_accessibility = [45, 47, 48, 50, 51, 52, 53, 54, 55, 57, 58, 59, 62, 63, 64, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 122, 125, 127, 129, 130, 150, 152, 153, 154, 156, 157, 158, 159]
        high_accessibility = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 49, 56, 60, 61, 65, 70, 71, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 110, 114, 118, 119, 120, 121, 123, 124, 126, 128, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 151, 155, 160, 161]

        # 计算高可达性区域 MPE 平均值
        high_accessibility_in_df2 = [x for x in high_accessibility if x in col_mpe_series.keys()]
        high_mpe_mean = col_mpe_series[high_accessibility_in_df2].mean()

        # 计算低可达性区域 MPE 平均值
        low_accessibility_in_df2 = [x for x in low_accessibility if x in col_mpe_series.keys()]
        low_mpe_mean = col_mpe_series[low_accessibility_in_df2].mean()

        # 计算结果
        result = high_mpe_mean - low_mpe_mean

        print(f"高可达性区域 MPE 平均值: {high_mpe_mean}")
        print(f"低可达性区域 MPE 平均值: {low_mpe_mean}")
        print(f"结果: {result}")


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
