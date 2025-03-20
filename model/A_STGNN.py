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
import math

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
# 定义ST-GNN模型
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, adj, x):
        # x shape: [batch, nodes, features, seq]
        # adj shape: [nodes, nodes]
        support = torch.matmul(x, self.weight)  # [batch, nodes, seq, out_feat]
        output = torch.matmul(adj, support)  # 空间特征聚合
        if self.bias is not None:
            output += self.bias
        return output


# 修正后的ST-GNN模型（文献[1][4]实现）
class STGNNBlock(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(STGNNBlock, self).__init__()
        self.gcn = GraphConvolution(in_features, out_features)
        self.tconv = nn.Conv2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=(K, 1),
            padding=(K // 2, 0)
        )
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # 输入维度: [batch, channels, nodes, seq_len]
        # 空间图卷积
        x = x.permute(0, 2, 3, 1)  # [batch, nodes, seq_len, features]
        x = self.gcn(adj, x)  # [batch, nodes, seq_len, out_features]
        x = x.permute(0, 3, 1, 2)  # [batch, out_features, nodes, seq_len]

        # 时间卷积
        x = self.tconv(x)  # [batch, out_features, nodes, seq_len]
        x = self.bn(x)
        return self.relu(x)


class STGNN(nn.Module):
    def __init__(self, num_nodes, time_steps, out_features):
        super(STGNN, self).__init__()
        hidden_features = 64
        K = 3

        self.block1 = STGNNBlock(1, hidden_features, K)
        self.block2 = STGNNBlock(hidden_features, hidden_features, K)

        # 计算全连接层输入维度（文献[4]方法）
        self.fc_input_dim = hidden_features * num_nodes * time_steps
        self.fc = nn.Linear(self.fc_input_dim, out_features)

    def forward(self, x, adj):
        # 输入x维度: [batch, 1, num_nodes, seq_len]
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = x.reshape(x.size(0), -1)  # [batch, hidden*num_nodes*seq_len]
        return self.fc(x)


# 初始化模型
num_nodes = 1
time_steps = hp.input_max_len
out_features = hp.future_seq
model = STGNN(num_nodes, time_steps, out_features).to(device)

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
