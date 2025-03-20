# @Time    : 2024/2/4 18:04
# @Author  : Gwy
# @Institution : IMU
# @IDE : PyCharm
# @FileName : 08_60m_Transformer
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

class TransformerModel(nn.Module):
    def __init__(self, hp):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(hp.hidden_size, hp.dropout_rate, hp.max_position_embeddings)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hp.hidden_size, nhead=hp.nhead,
                                                    dim_feedforward=hp.dim_feedforward,
                                                    dropout=hp.dropout_rate,
                                                    activation=hp.activation_function, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=hp.num_layers)
        self.encoder = nn.Linear(hp.input_size, hp.hidden_size)
        self.decoder = nn.Linear(hp.hidden_size, hp.output_size)

    def forward(self, x):
        x = self.encoder(x)  # 将输入维度转换为 hidden_size
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x

# 使用Hyperparams实例化模型
model = TransformerModel(hp).to(device)


# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

# 训练模型
for epoch in range(hp.num_epochs):
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)  # 将数据移到 GPU
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 每 10 轮打印一次平均损失
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{hp.num_epochs}, Average Loss: {avg_loss:.4f}")


# 测试模型
model.eval()
results = []

for _ in range(10):
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
df.to_excel('./each_step_metrics_pems08/Transformer_pytorch_60m.xlsx', index=False)

# 可选：打印评估结果
print(df)

# 绘制图形
# 注意：这里只绘制最后一次预测的结果
sing_pred = predicts[:,0]
sing_label = test_label[:,0]
#draw_gt_pred(sing_label, sing_pred)
#draw_gt_pred(sing_label[288:288*2], sing_pred[288:288*2])
