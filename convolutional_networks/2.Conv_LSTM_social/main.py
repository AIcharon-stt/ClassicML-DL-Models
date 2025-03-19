# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/9 13:02
# ------------------------------------------------------------------------------

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from Model import Conv_RNN
from Function import preprocess_data, train_model, evaluate_model, plot_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 检查并创建结果文件夹
if not os.path.exists('result'):
    os.makedirs('result')


def main():
    # 参数设置
    params = {
        'sequence_length': 6,
        'hidden_layer_size': 64,
        'num_channels': 32,  # 卷积层通道数
        'batch_size': 64,
        'num_epochs': 400,
        'learning_rate': 0.001,
        'num_layers': 4,
        'model_type': 'Conv_GRU',  # 可以设置为 'Conv_RNN', 'Conv_LSTM', 'Conv_GRU'
        'selected_columns': ['ID', 'POID','TransportationD','IndustrialD']
        # 可以设置为 'Area', 'poptotalD', 'less14D', 'less64D', 'more64D',
        # 'localpopD', 'ParkD', 'IndustrialD', 'TransportationD', 'ResidenceD', 'CommercialD', 'PublicD', 'POID',
        # 'Diversity', 'PGDP', 'Aprice', 'RoadDensity', 'Distancetocitycenter', 'Residential ratio', 'Commercial
        # ratio', 'Industrial ratio'
    }

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    demand_data = pd.read_csv(r'data\TNC_demand_data.csv', encoding='gbk')
    feature_data = pd.read_csv(r'data\Feature_data.csv', encoding='gbk')
    social_data = pd.read_csv(r'data\Social_factor.csv', encoding='gbk')  # 假设第一列是索引

    social_data_column = social_data[params['selected_columns']]

    # 确保两个数据集行数相同
    if demand_data.shape[0] != feature_data.shape[0]:
        raise ValueError("Row counts of demand_data and feature_data do not match.")

    # 整合数据
    combined_data = pd.concat([demand_data, feature_data], axis=1)
    combined_data.to_csv('1.csv')

    # 数据预处理
    x, y, social_features, scaler_time, scaler_social = preprocess_data(combined_data, social_data_column,
                                                                        params['sequence_length'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    social_train, social_test = train_test_split(social_features, test_size=0.2, random_state=42)

    # 转换为PyTorch张量
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    social_train = torch.tensor(social_train, dtype=torch.float32).to(device)
    social_test = torch.tensor(social_test, dtype=torch.float32).to(device)

    # 初始化模型
    model = Conv_RNN(params['model_type'], combined_data.shape[1], x_train.shape[2], params['hidden_layer_size'],
                     params['num_layers'], params['num_channels'], social_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 训练模型
    train_losses = train_model(model, optimizer, params['num_epochs'], x_train, y_train, social_train)

    # 绘制损失函数曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('result/loss_curve.png')
    plt.close()

    # 评估模型
    evaluated_metrics, predictions, y_test_original = evaluate_model(model, x_test, y_test, social_test, scaler_time,
                                                                     params['model_type'])

    print("Evaluated Metrics:", evaluated_metrics)

    # 绘制预测和真实值拟合曲线
    plot_predictions(predictions, y_test_original, 'Predictions vs Actual Data')


if __name__ == "__main__":
    main()
