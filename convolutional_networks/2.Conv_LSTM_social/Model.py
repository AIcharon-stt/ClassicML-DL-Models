# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/9 13:07
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Conv_RNN 模型定义
class Conv_RNN(nn.Module):
    def __init__(self, model_type, num_classes, input_size, hidden_size, num_layers, num_channels, social_feature_size):
        super(Conv_RNN, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 卷积层
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=3, padding=1)

        # RNN 层
        if model_type == 'Conv_LSTM':
            self.rnn = nn.LSTM(num_channels, hidden_size, num_layers, batch_first=True)
        elif model_type == 'Conv_GRU':
            self.rnn = nn.GRU(num_channels, hidden_size, num_layers, batch_first=True)
        else:  # Conv_RNN
            self.rnn = nn.RNN(num_channels, hidden_size, num_layers, batch_first=True)

        # 非时间序列特征处理层
        self.fc_social = nn.Linear(social_feature_size, hidden_size)

        # 最终输出层
        self.fc_final = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x, social_features):
        # 卷积层
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)

        # RNN 层
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if self.model_type == 'Conv_LSTM':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        out = out[:, -1, :]

        # 非时间序列特征处理
        social_out = F.relu(self.fc_social(social_features))

        # 合并两类特征
        combined = torch.cat((out, social_out), dim=1)

        # 最终输出层
        out = self.fc_final(combined)
        return out
