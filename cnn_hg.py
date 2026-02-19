import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, 3)
        self.conv2 = nn.Conv1d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv1d(32, 32, 3, 1, 1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(352, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # 第一层卷积 + ReLU 激活 + 池化
        x = self.pool(torch.relu(self.conv1(x)))  # 90
        # 第二层卷积 + ReLU 激活 + 池化
        x = self.pool(torch.relu(self.conv2(x)))  # 22
        # 第三层卷积 + ReLU 激活 + 池化
        x = self.pool(torch.relu(self.conv3(x)))  # 32*11
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 全连接层 + ReLU 激活
        x = torch.relu(self.fc1(x))
        # 输出层
        x = torch.softmax(self.fc2(x), dim=1)
        return x


# 加载.pt文件
data_dict = torch.load('dataset_split.pt')

# 取出训练集和测试集
X_train = data_dict['X_train']  # 训练集特征
y_train = data_dict['y_train']  # 训练集标签
X_test = data_dict['X_test']    # 测试集特征
y_test = data_dict['y_test']    # 测试集标签

# 打印形状验证
print("训练集特征:", X_train.shape)
print("训练集标签:", y_train.shape)
print("测试集特征:", X_test.shape)
print("测试集标签:", y_test.shape)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 10
learning_rate = 0.001
num_epochs = 500

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差（MSE）损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型

# 存储每个 epoch 的指标
all_epoch_loss = []  # 均方误差 MSE 即训练使用的损失函数
all_epoch_r2 = []  # 决定系数r2
all_epoch_mae = []  # 计算 MAE（平均绝对误差）
all_epoch_rmse = []  # 计算 RMSE（均方根误差）

# 存储每个 epoch 的指标
all_test_epoch_loss = []
all_test_epoch_r2 = []
all_test_epoch_mae = []
all_test_epoch_rmse = []

for epoch in range(num_epochs):
    model.train()  # 将模型设置成训练模式
    running_loss = 0.0

    # 用于计算指标
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # running_loss += loss.item()
        # 计算总损失
        running_loss += loss.item() * inputs.size(0)

        # 收集预测值和真实值
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    # 计算每个 epoch 的平均损失
    epoch_loss = running_loss / len(train_loader.dataset)  # !!!
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    all_epoch_loss.append(epoch_loss)

    # 将预测值和真实值拼接在一起
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算 R^2（决定系数）
    ss_total = np.sum((all_labels - np.mean(all_labels)) ** 2)
    ss_residual = np.sum((all_labels - all_preds) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    all_epoch_r2.append(r2)
    # 计算 MAE（平均绝对误差）
    mae = np.mean(np.abs(all_labels - all_preds))
    all_epoch_mae.append(mae)
    # 计算 RMSE（均方根误差）
    rmse = np.sqrt(np.mean((all_labels - all_preds) ** 2))
    all_epoch_rmse.append(rmse)

    ################
    # 计算此时测试集的指标
    ################
    model.eval()  # 将模型设置为评估模式，禁用 dropout 等
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            # 打印标签和预测值
            # print(f"真实标签: {labels}, 预测标签: {outputs}")

            # 将预测结果和真实标签收集到列表中
            all_test_preds.append(outputs.detach().cpu().numpy())
            all_test_labels.append(labels.detach().cpu().numpy())

    # 将所有预测值和标签合并
    all_test_preds = np.concatenate(all_test_preds, axis=0)
    all_test_labels = np.concatenate(all_test_labels, axis=0)

    # 计算 R² (决定系数)
    test_ss_total = np.sum((all_test_labels - np.mean(all_test_labels)) ** 2)
    test_ss_residual = np.sum((all_test_labels - all_test_preds) ** 2)
    test_r2 = 1 - (test_ss_residual / test_ss_total)
    # 计算 MAE (平均绝对误差)
    test_mae = np.mean(np.abs(all_test_labels - all_test_preds))
    # 计算 RMSE (均方根误差)
    test_rmse = np.sqrt(np.mean((all_test_labels - all_test_preds) ** 2))
    # 计算 MSE (均方误差)
    test_mse = np.mean((all_test_labels - all_test_preds) ** 2)

    all_test_epoch_loss.append(test_mse)
    all_test_epoch_r2.append(test_r2)
    all_test_epoch_mae.append(test_mae)
    all_test_epoch_rmse.append(test_rmse)

# 保存模型参数
torch.save(model.state_dict(), './CNN.pth')

# 训练集
txt = "./CNN_train.txt"
np.savetxt(txt, np.column_stack((all_epoch_loss, all_epoch_r2, all_epoch_mae, all_epoch_rmse)), fmt="%.8f",
           comments='')
# 测试集
txt = "./CNN_test.txt"
np.savetxt(txt, np.column_stack((all_test_epoch_loss, all_test_epoch_r2, all_test_epoch_mae, all_test_epoch_rmse)),
           fmt="%.8f",
           comments='')

####################
# 训练完成后模型的预测结果
####################
all_test_preds = []
all_test_labels = []
all_train_preds = []
all_train_labels = []
model.eval()  # 将模型设置为评估模式，禁用 dropout 等
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)

        # 打印标签和预测值
        # print(f"真实标签: {labels}, 预测标签: {outputs}")

        # 将预测结果和真实标签收集到列表中
        all_test_preds.append(outputs.detach().cpu().numpy())
        all_test_labels.append(labels.detach().cpu().numpy())

    for inputs, labels in train_loader:
        outputs = model(inputs)

        # 打印标签和预测值
        # print(f"真实标签: {labels}, 预测标签: {outputs}")

        # 将预测结果和真实标签收集到列表中
        all_train_preds.append(outputs.detach().cpu().numpy())
        all_train_labels.append(labels.detach().cpu().numpy())

# 将预测值和标签合并
all_test_preds = np.concatenate(all_test_preds, axis=0)
all_test_labels = np.concatenate(all_test_labels, axis=0)

all_train_preds = np.concatenate(all_train_preds, axis=0)
all_train_labels = np.concatenate(all_train_labels, axis=0)

txt = "./CNN_test_prediction.txt"
np.savetxt(txt, np.column_stack((all_test_preds, all_test_labels)),
           fmt="%.8f", comments='')
txt = "./CNN_train_prediction.txt"
np.savetxt(txt, np.column_stack((all_train_preds, all_train_labels)),
           fmt="%.8f", comments='')


# 计算测试集上的指标
test_mse = np.mean((all_test_labels - all_test_preds) ** 2)
test_r2 = 1 - (np.sum((all_test_labels - all_test_preds) ** 2) / np.sum(
    (all_test_labels - np.mean(all_test_labels)) ** 2))
test_mae = np.mean(np.abs(all_test_labels - all_test_preds))
test_rmse = np.sqrt(test_mse)
test_avg_error = np.mean(np.abs(all_test_labels - all_test_preds))

# 计算训练集上的指标
train_mse = np.mean((all_train_labels - all_train_preds) ** 2)
train_r2 = 1 - (np.sum((all_train_labels - all_train_preds) ** 2) / np.sum(
    (all_train_labels - np.mean(all_train_labels)) ** 2))
train_mae = np.mean(np.abs(all_train_labels - all_train_preds))
train_rmse = np.sqrt(train_mse)
train_avg_error = np.mean(np.abs(all_train_labels - all_train_preds))

dd = [[train_mse, train_r2, train_mae, train_rmse, train_avg_error],
      [test_mse, test_r2, test_mae, test_rmse, test_avg_error]]

# 保存指标
txt = "./CNN_model_test.txt"
np.savetxt(txt, dd, fmt='%.6f', delimiter=',', header='MSE,R2,MAE,RMSE,Avg Error', comments='')

plt.figure()
plt.plot(all_epoch_loss, color='#BF1D2D', linewidth=1, label='epoch loss')
plt.plot(all_test_epoch_loss, color='#262626', linewidth=1, label='test epoch loss')
plt.title("MSE")
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()

plt.figure()
plt.plot(all_epoch_r2, color='#BF1D2D', linewidth=1, label='epoch loss')
plt.plot(all_test_epoch_r2, color='#262626', linewidth=1, label='test epoch loss')
plt.title("R2")
plt.xlabel('epoch')
plt.ylabel('R2')
plt.legend()
plt.show()