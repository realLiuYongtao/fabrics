import numpy as np
import torch
from sklearn.model_selection import train_test_split

d_num = 232
j_num = 40
m_num = 128
h_num = 208

Data = []
Lable = []

for i in range(1, d_num + 1):
    txt = "median_filter_data/5/d/d_{}.txt".format(i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
    Lable.append([1, 0, 0])
for i in range(1, m_num + 1):
    txt = "median_filter_data/5/m/m_{}.txt".format(i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
    Lable.append([0, 1, 0])
for i in range(1, j_num + 1):
    txt = "median_filter_data/5/j/j_{}.txt".format(i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
    Lable.append([0, 0, 1])
for i in range(1, h_num + 1):
    txt = "median_filter_data/5/h/h_{}.txt".format(i)
    data = np.loadtxt(txt)
    y = data[:, 0]
    Data.append(y)
txt = "h/lable.txt"
data = np.loadtxt(txt)
print(data)
l1 = data[:, 0]
l2 = data[:, 1]
l3 = data[:, 2]
for i in range(len(l1)):
    for j in range(1, 9):
        Lable.append(np.array([l1[i], l2[i], l3[i]], dtype=np.float32))

# 转换为 NumPy 数组
Data = np.array(Data, dtype=np.float32)  # 数据
Lable = np.array(Lable, dtype=np.float32)  # 标签

# 假设 Data 是形状为 (280, 544) 的 NumPy 数组
Data = torch.tensor(Data, dtype=torch.float32).unsqueeze(1)  # 将形状改为 (280, 1, 544)

print("数据集大小:", Data.shape)
print("标签集大小:", Lable.shape)

# 划分数据集，70% 训练集，30% 测试集
X_train, X_test, y_train, y_test = train_test_split(Data, Lable, test_size=0.3, random_state=42)

# 保存训练集和测试集
# 方法1：保存为PyTorch的.pt文件
torch.save({
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test
}, 'dataset_split.pt')

