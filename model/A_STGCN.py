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

# 设定工作目录
os.chdir('../data')

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

# 定义邻接矩阵（这里使用单位矩阵作为示例）
num_nodes = train_x.shape[2]  # 图中节点数
adj = torch.eye(num_nodes).to(device)

# 设置STGCN模型参数
class STGCNBlock(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(STGCNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, K), padding=(0, K//2))
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=(1, K), padding=(0, K//2))
        if in_features != out_features:
            self.adjust_channels = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1))
        else:
            self.adjust_channels = None

    def forward(self, x, adj):
        if self.adjust_channels:
            x = self.adjust_channels(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        #print(f"Shape after conv1: {x.shape}")
        x = x.permute(0, 1, 3, 2)
        x = self.conv2(x)
        #print(f"Shape after conv2: {x.shape}")
        x = x.permute(0, 3, 2, 1)
        return x


class STGCN(nn.Module):
    def __init__(self, num_nodes, time_steps, out_features):
        super(STGCN, self).__init__()
        in_features = 1
        hidden_features = 64
        K = 3
        self.stgcn1 = STGCNBlock(in_features, hidden_features, K)
        self.stgcn2 = STGCNBlock(hidden_features, hidden_features, K)

        # 正确计算全连接层的输入特征数
        # 根据卷积后的实际输出尺寸调整全连接层的输入
        self.fc_input_features = hidden_features * num_nodes * time_steps * 64  # 由于最后的形状是 [batch, 64, 6, 64]
        self.fc = nn.Linear(self.fc_input_features, out_features)

    def forward(self, x, adj):
        x = self.stgcn1(x, adj)
        x = self.stgcn2(x, adj)
        x = x.reshape(x.size(0), -1)
        #print(f"Flattened size: {x.size()}")  # 调试打印展平后的尺寸
        x = self.fc(x)
        return x


# 初始化模型
num_nodes = 1
time_steps = hp.input_max_len
out_features = hp.future_seq
model = STGCN(num_nodes, time_steps, out_features).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

# 训练模型
for epoch in range(hp.num_epochs):
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)  # 添加通道维度
        outputs = model(x, adj)
        y = y.squeeze(-1)
        y = y.view(y.size(0), -1)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{hp.num_epochs}, Average Loss: {avg_loss:.4f}")

# 测试模型
model.eval()
results = []

with torch.no_grad():
    test_input_tensor = test_input_tensor.to(device)
    test_input_tensor = test_input_tensor.unsqueeze(1)
    predicts_tensor = model(test_input_tensor, adj)
    predicts = sca.inverse_transform(predicts_tensor.cpu().numpy().squeeze())

    mae, mse, rmse, mape = evaluate_model(predicts, test_label)
    results.append({'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape})

df = pd.DataFrame(results)
os.chdir('..')
print(df)
