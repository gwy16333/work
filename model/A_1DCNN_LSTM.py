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
# 使用同样的 utils 和 parameters 文件
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

# 定义CNN-LSTM模型
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hp.input_size, out_channels=hp.out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hp.out_channels, out_channels=hp.out_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=hp.out_channels * 2, hidden_size=hp.hidden_size, num_layers=hp.num_layers, batch_first=True)
        self.fc = nn.Linear(hp.hidden_size, hp.output_size)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, input_size, seq_length]
        #print(f"After permute: {x.shape}")
        x = self.cnn(x)  # 通过CNN层
        #print(f"After CNN: {x.shape}")
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, seq_length, out_channels*2]
        #print(f"After permute back: {x.shape}")
        x, _ = self.lstm(x)  # 通过LSTM层
        #print(f"After LSTM: {x.shape}")
        x = self.fc(x)  # 通过全连接层，输出形状：[batch_size, seq_length, output_size]
        #print(f"After FC: {x.shape}")
        return x

model = CNN_LSTM().to(device)  # 将模型移到 GPU

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
        #print(f"Output shape: {outputs.shape}, Target shape: {y.shape}")  # 调试输出形状
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 增加调试信息，输出当前批次的损失
        #if (i + 1) % 10 == 0:
            #print(f"Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

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
#df.to_excel('../each_step_metrics_pems08/cnn_lstm_pytorch_60m.xlsx', index=False)

# 可选：打印评估结果
print(df)

# 绘制图形
# 注意：这里只绘制最后一次预测的结果
sing_pred = predicts[:, 0]
sing_label = test_label[:, 0]
#draw_gt_pred(sing_label, sing_pred)
#draw_gt_pred(sing_label[288:288*2], sing_pred[288:288*2])
