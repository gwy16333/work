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

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
# 在模型中应用
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding))

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)  # 在通道维度上连接

class MultiHeadMultiScaleAttention(nn.Module):
    def __init__(self, channel_size, head_kernel_sizes):
        super(MultiHeadMultiScaleAttention, self).__init__()
        self.heads = nn.ModuleList()
        for kernel_sizes in head_kernel_sizes:
            scales = nn.ModuleList([nn.Conv1d(channel_size, channel_size, kernel_size=k, padding=k//2) for k in kernel_sizes])
            self.heads.append(scales)

    def forward(self, x):
        head_outputs = []
        for scales in self.heads:
            attention_maps = [torch.sigmoid(scale(x)) for scale in scales]
            out = sum([x * att for att in attention_maps])
            head_outputs.append(out)
        return torch.cat(head_outputs, dim=1)


class CNNEncoder(nn.Module):
    def __init__(self, hp):
        super(CNNEncoder, self).__init__()
        # 使用MultiScaleConvBlock代替原有的单层卷积
        self.multi_scale_conv_block = MultiScaleConvBlock(
            in_channels=hp.input_size,
            out_channels=hp.out_channels,
            kernel_sizes=hp.kernel_sizes  # 假设hp对象有kernel_sizes属性
        )
        self.dropout = nn.Dropout(hp.cnn_dropout_rate)

    def forward(self, x):
        x = self.multi_scale_conv_block(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class CNNTransformerModel(nn.Module):
    def __init__(self, input_size, out_channels, kernel_sizes, head_kernel_sizes, num_heads, hidden_size, output_size, max_len, dropout_rate):
        super(CNNTransformerModel, self).__init__()
        self.cnn_encoder = MultiScaleConvBlock(input_size, out_channels, kernel_sizes)
        self.multi_scale_attention = MultiHeadMultiScaleAttention(out_channels * len(kernel_sizes), head_kernel_sizes)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_rate, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.fc1 = nn.Linear(out_channels * len(kernel_sizes) * len(head_kernel_sizes), hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert (batch_size, seq_len, features) to (batch_size, features, seq_len) for CNN
        x = self.cnn_encoder(x)
        x = self.multi_scale_attention(x)
        x = x.transpose(1, 2)  # Convert back to (batch_size, seq_len, features)
        x = self.fc1(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc2(x)
        return x


model = CNNTransformerModel(
    input_size=hp.input_size,
    out_channels=hp.out_channels,
    kernel_sizes=hp.kernel_sizes,
    head_kernel_sizes=[hp.kernel_sizes for _ in range(hp.nhead)],  # 假设每个头使用相同的卷积核大小列表
    num_heads=hp.nhead,
    hidden_size=hp.hidden_size,
    output_size=hp.output_size,
    max_len=hp.max_position_embeddings,
    dropout_rate=hp.dropout_rate
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
