# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/9 20:40
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data():
    demand_data = pd.read_csv(r'data_162\TNC_demand_data.csv')
    feature_data = pd.read_csv(r'data_162\Feature_data.csv')
    return demand_data, feature_data


def preprocess_data(demand_data, feature_data):
    data = pd.concat([demand_data, feature_data], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_data.iloc[:, demand_data.shape[1]:])     #  选取训练数据中除了目标变量之外的特征列，然后对这些特征进行标准化。
    test_features = scaler.transform(test_data.iloc[:, demand_data.shape[1]:])

    return train_data, test_data, train_features, test_features


class DemandDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_dataloader(features, labels, batch_size):
    dataset = DemandDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = y_true > 0.1
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))*100
        mpe = np.mean((y_true[mask] - y_pred[mask]) / y_true[mask])*100
    else:
        mape = 0
        mpe = 0

    print(f"MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}, MPE: {mpe}, R^2: {r2}")

    return {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape, 'MPE': mpe, 'R2': r2, 'MPE_values': 100*(y_true[mask] - y_pred[mask]) / y_true[mask]}, mpe
