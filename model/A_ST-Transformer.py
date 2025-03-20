# @Time    : 2024/2/4 18:04
# @Author  : Gwy
# @Institution : IMU
# @IDE : PyCharm
# @FileName : CNN_Transformer1_08_60m
# @Project Name :code

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from Config import Hyperparams as hp
from data_utils import *
import torch.nn.functional as F
import os
os.chdir('../data')
# 使用同样的 utilss_ 和 parameters 文件
# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取和处理数据
datas = load_data(hp.pkl_path_08)
train_x = np.array(datas[0])[:, :hp.maxlen]
train_y = np.array(datas[0])[:, -hp.output_max_len:]


test_data = np.array(datas[2])
sca = datas[-1]
test_label = sca.inverse_transform(np.squeeze(test_data[:, -hp.output_max_len:]))
test_input = test_data[:, :hp.output_max_len]


train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
test_input_tensor = torch.tensor(test_input, dtype=torch.float32)


# 创建 DataLoader
train_data = TensorDataset(train_x_tensor, train_y_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=hp.batch_size, shuffle=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


class STTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # 空间注意力（特征维度）
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_spatial = nn.Dropout(dropout)

        # 时间注意力（时间维度）
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_temporal = nn.Dropout(dropout)

        # 前馈网络
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x):
        # 空间注意力
        residual = x
        x = self.spatial_norm(x)
        x, _ = self.spatial_attn(x, x, x)
        x = residual + self.dropout_spatial(x)

        # 时间注意力
        residual = x
        x = self.temporal_norm(x)
        x, _ = self.temporal_attn(x, x, x)
        x = residual + self.dropout_temporal(x)

        # 前馈网络
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout_ffn(x)
        return x


class STTransformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, output_size, max_len, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([STTransformerLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x形状: [batch, seq_len, input_size]
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        output = self.output_layer(x)
        return output[:, -hp.output_max_len:, :]  # 取最后预测的时间步


# 初始化模型
model = STTransformer(
    input_size=hp.input_size,
    d_model=hp.hidden_size,
    num_heads=hp.nhead,
    num_layers=hp.num_layers,
    output_size=hp.output_size,
    max_len=hp.max_position_embeddings,
    dropout=hp.dropout_rate
).to(device)





# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 100))

def single_mask(tensor, mask_prob=0.1):
    """
    对给定的张量应用单个掩码。
    :param tensor: 输入的Tensor。
    :param mask_prob: 掩码的概率。
    :return: 掩码后的Tensor。
    """
    mask = (torch.rand(tensor.size()) > mask_prob).float().to(tensor.device)
    return tensor * mask

# 在训练循环中应用 single_mask
for epoch in range(hp.num_epochs):
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)  # 将数据移到 GPU
        masked_x = single_mask(x)  # 应用掩码
        optimizer.zero_grad()
        outputs = model(masked_x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # 更新学习率
    scheduler.step()

    # 每 10 轮打印一次平均损失
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{hp.num_epochs}, Average Loss: {avg_loss:.4f}")


# 测试模型
model.eval()
results = []


with torch.no_grad():
    test_input_tensor = test_input_tensor.to(device)
    predicts_tensor = model(test_input_tensor)
        # 将预测结果移回 CPU
    predicts = sca.inverse_transform(predicts_tensor.cpu().numpy().squeeze())

    # 评估
mae, mse, rmse, mape = evaluate_model(predicts, test_label)

    # 添加到结果列表
results.append({'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape})

# 将结果保存到DataFrame中
df = pd.DataFrame(results)
os.chdir('..')
# 将结果保存到Excel文件中
df.to_excel('./each_step_metrics_pems08/CNN+Transformer_pytorch_60m.xlsx', index=False)

# 可选：打印评估结果
print(df)

# 绘制图形
# 注意：这里只绘制最后一次预测的结果
sing_pred = predicts[:,0]
sing_label = test_label[:,0]
#draw_gt_pred(sing_label, sing_pred)
#draw_gt_pred(sing_label[288:288*2], sing_pred[288:288*2])
