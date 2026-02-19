import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(1,16,7,3)
        self.conv2 = nn.Conv1d(16,32,3,2,1)
        self.conv3 = nn.Conv1d(32,32,3,1,1)
        self.pool = nn.MaxPool1d(3,2)
        self.fc1 = nn.Linear(352,64)
        self.fc2 = nn.Linear(64,3)

    def forward(self, x):
        # 第一层卷积 + ReLU 激活 + 池化
        x = self.pool(torch.relu(self.conv1(x))) # 90
        # 第二层卷积 + ReLU 激活 + 池化
        x = self.pool(torch.relu(self.conv2(x))) # 22
        # 第三层卷积 + ReLU 激活 + 池化
        x = self.pool(torch.relu(self.conv3(x))) # 32*11
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 全连接层 + ReLU 激活
        x = torch.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x



median_filter = 3  #3  5  7  9
d_num = 232
j_num = 40
m_num = 128
h_num = 144

Data = []
Lable = []

for i in range(1, d_num + 1):
    txt = "median_filter_data/{}/d/d_{}.txt".format(median_filter,i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
    Lable.append([1,0,0])
for i in range(1, m_num + 1):
    txt = "median_filter_data/{}/m/m_{}.txt".format(median_filter,i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
    Lable.append([0,1,0])
for i in range(1, j_num + 1):
    txt = "median_filter_data/{}/j/j_{}.txt".format(median_filter,i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
    Lable.append([0,0,1])
for i in range(1, h_num + 1):
    txt = "median_filter_data/{}/h/h_{}.txt".format(median_filter,i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
txt = "h/lable.txt"
data = np.loadtxt(txt)
print(data)
l1 = data[:, 0]
l2 = data[:, 1]
for i in range(len(l1)):
    for j in range(1,9):
        Lable.append(np.array([l1[i],l2[i],0.0],dtype=np.float32))


# 转换为 NumPy 数组
Data = np.array(Data)  # 数据
Lable = np.array(Lable)  # 标签

# 将标签从独热编码转换为类别索引
# 对每一行取最大值的索引（0, 1, 2）
Lable = np.argmax(Lable, axis=1)


# 假设 Data 是形状为 (280, 544) 的 NumPy 数组
Data = torch.tensor(Data, dtype=torch.float32).unsqueeze(1)  # 将形状改为 (280, 1, 544)

print("数据集大小:", Data.shape)
print("标签集大小:", Lable.shape)

# 划分数据集，70% 训练集，30% 测试集
X_train, X_test, y_train, y_test = train_test_split(Data, Lable, test_size=0.3, random_state=42)

# 输出训练集和测试集的形状
print("训练集大小:", X_train.shape, y_train.shape)
print("测试集大小:", X_test.shape, y_test.shape)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
learning_rate = 0.001
num_epochs = 10

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
model = Net()



# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        # 计算准确度
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# 保存模型参数
torch.save(model.state_dict(), 'model.pth')




# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 打印标签和预测值
        print(f"真实标签: {labels}, 预测标签: {predicted}")

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')
