# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/9 20:38
# ------------------------------------------------------------------------------
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
import torch.nn as nn


# （HA）
def historical_average(train_labels):
    return np.mean(train_labels)


# ARIMA
def arima_model(train_labels, order=(1, 1, 1)):
    model = ARIMA(train_labels, order=order)
    model_fit = model.fit()
    return model_fit


# SVR
class SVRModel(nn.Module):
    def __init__(self):
        super(SVRModel, self).__init__()
        self.svr = SVR()

    def fit(self, X, y):
        self.svr.fit(X, y)

    def predict(self, X):
        return self.svr.predict(X)


def train_and_predict(model_type, train_features, train_labels, test_features):
    if model_type == 'HA':
        ha_pred = historical_average(train_labels)
        test_predictions = np.full(shape=len(test_features), fill_value=ha_pred)
    elif model_type == 'ARIMA':
        arima_model_fit = arima_model(train_labels)
        test_predictions = arima_model_fit.forecast(steps=len(test_features))
    elif model_type == 'SVR':
        svr_model = SVRModel()
        svr_model.fit(train_features, train_labels)
        test_predictions = svr_model.predict(test_features)
    else:
        raise ValueError("Invalid model type. Choose 'HA', 'ARIMA', or 'SVR'.")

    return test_predictions
