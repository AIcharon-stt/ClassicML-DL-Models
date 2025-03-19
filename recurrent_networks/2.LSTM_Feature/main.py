# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/9 13:02
# ------------------------------------------------------------------------------

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
from Model import RNNModel
from Function import preprocess_data, train_model, evaluate_model, plot_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 检查并创建结果文件夹
if not os.path.exists('result'):
    os.makedirs('result')


def main():
    # 参数设置
    params = {
        'sequence_length': 5,
        'hidden_layer_size': 100,
        'batch_size': 64,
        'num_epochs': 300,
        'learning_rate': 0.001,
        'num_layers': 2,
        'model_type': 'LSTM'  # 可以设置为 'RNN', 'LSTM', 'GRU'
    }

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    demand_data = pd.read_csv(r'data\TNC_demand_data.csv', encoding='gbk')
    feature_data = pd.read_csv(r'data\Feature_data.csv', encoding='gbk')

    # 确保两个数据集行数相同
    if demand_data.shape[0] != feature_data.shape[0]:
        raise ValueError("demand_data and feature_data 两个数据集行数不一致.")

    # 初始化评价指标的存储
    metrics_summary = {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': [], 'MAPE': [], 'MPE': []}
    mpe_values_list = []
    mpe_values = []

    for column in range(demand_data.shape[1]):
        print(f"训练地区 {column + 1}/{demand_data.shape[1]}")
        # 数据预处理
        x, y, scaler_combined, scaler_demand = preprocess_data(demand_data.iloc[:, [column]], feature_data,
                                                               params['sequence_length'])
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        # 转换为PyTorch张量
        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # 初始化模型
        model = RNNModel(params['model_type'], 1, x_train.shape[2], params['hidden_layer_size'],
                         params['num_layers']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # 训练模型
        train_losses = train_model(model, optimizer, params['num_epochs'], x_train, y_train)

        # 绘制并保存loss图
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.title(f'Loss over Epochs for Column {column}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'result/loss_column_{column}.png')
        plt.close()

        # 评估模型
        evaluated_metrics, predictions, y_test_original, mpe = evaluate_model(model, x_test, y_test, scaler_demand)
        mpe_values.append(mpe)
        for key in metrics_summary:
            metrics_summary[key].append(evaluated_metrics[key])

        mpe_values_list.append(evaluated_metrics['MPE_values'])

        # 绘制并保存预测与真实值拟合曲线图
        plot_predictions(predictions, y_test_original[:, 0], column)

        print(f"地区 {column + 1} 测试评价指标:: {evaluated_metrics}")

    # 计算所有列的评价指标平均值
    average_metrics = {key: np.mean(values) for key, values in metrics_summary.items()}
    print("所有地区的评价指标均值:", average_metrics)

    # Write mpe values for each iteration to a CSV file with transposition and column index starting from 0
    pd.DataFrame(mpe_values, columns=['MPE']).transpose().reset_index(drop=True).to_csv('Results/mpe_values_average_LSTM_Feature.csv', index=False)

    # Write MPE values to a CSV file with transposition and column index starting from 0
    mpe_values_df = pd.DataFrame(mpe_values_list).transpose().reset_index(drop=True)
    mpe_values_df.columns = range(demand_data.shape[1])  # Set column index starting from 0
    mpe_values_df.to_csv('Results/mpe_values_LSTM_Feature.csv', index=False)


if __name__ == "__main__":
    main()
