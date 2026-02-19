import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


median_filter = [1,3,5,7,9,11,13,15,17]
lr = [0,1,2,3,4,5,6,7,8,9,10,20,30]


"""
# ##############################################################
# 训练集
# ##############################################################

median_filter_1 = np.loadtxt("result/cnn_hg_median_filter_1.txt")
median_filter_3 = np.loadtxt("result/cnn_hg_median_filter_3.txt")
median_filter_5 = np.loadtxt("result/cnn_hg_median_filter_5.txt")
median_filter_7 = np.loadtxt("result/cnn_hg_median_filter_7.txt")
median_filter_9 = np.loadtxt("result/cnn_hg_median_filter_9.txt")
median_filter_11 = np.loadtxt("result/cnn_hg_median_filter_11.txt")
median_filter_13 = np.loadtxt("result/cnn_hg_median_filter_13.txt")
median_filter_15 = np.loadtxt("result/cnn_hg_median_filter_15.txt")
median_filter_17 = np.loadtxt("result/cnn_hg_median_filter_17.txt")


all_epoch_loss = [] # 均方误差 MSE 即训练使用的损失函数
all_epoch_r2 = [] # 决定系数 R-squared
all_epoch_mae = [] # 计算 MAE（平均绝对误差）
all_epoch_rmse = [] # 计算 RMSE（均方根误差）


# 均方误差 MSE
plt.figure()
plt.plot(median_filter_1[:, 0], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(median_filter_3[:, 0], color='#7d1e3f', linewidth=1, label='median_filter_3')
plt.plot(median_filter_5[:, 0], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(median_filter_7[:, 0], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(median_filter_9[:, 0], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(median_filter_11[:, 0], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(median_filter_13[:, 0], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(median_filter_15[:, 0], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(median_filter_17[:, 0], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("train MSE")
plt.xlabel('epoch')
plt.ylabel('train MSE')
# 显示图例
plt.legend()

# 决定系数 R-squared
plt.figure()
plt.plot(median_filter_1[:, 1], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(median_filter_3[:, 1], color='#7d1e3f', linewidth=1, label='median_filter_3')
plt.plot(median_filter_5[:, 1], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(median_filter_7[:, 1], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(median_filter_9[:, 1], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(median_filter_11[:, 1], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(median_filter_13[:, 1], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(median_filter_15[:, 1], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(median_filter_17[:, 1], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("train R-squared")
plt.xlabel('epoch')
plt.ylabel('train R-squared')
# 显示图例
plt.legend()



# 平均绝对误差 RMSE
plt.figure()
plt.plot(median_filter_1[:, 2], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(median_filter_3[:, 2], color='#7d1e3f', linewidth=1, label='median_filter_3')
plt.plot(median_filter_5[:, 2], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(median_filter_7[:, 2], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(median_filter_9[:, 2], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(median_filter_11[:, 2], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(median_filter_13[:, 2], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(median_filter_15[:, 2], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(median_filter_17[:, 2], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("train MAE")
plt.xlabel('epoch')
plt.ylabel('train MAE')
# 显示图例
plt.legend()


# 均方根误差 MAE
plt.figure()
plt.plot(median_filter_1[:, 3], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(median_filter_3[:, 3], color='#7d1e3f', linewidth=1, label='median_filter_3')
plt.plot(median_filter_5[:, 3], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(median_filter_7[:, 3], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(median_filter_9[:, 3], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(median_filter_11[:, 3], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(median_filter_13[:, 3], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(median_filter_15[:, 3], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(median_filter_17[:, 3], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("train RMSE")
plt.xlabel('epoch')
plt.ylabel('train RMSE')
# 显示图例
plt.legend()



# ##############################################################
# 测试集
# ##############################################################

predict_median_filter_1 = np.loadtxt("result/cnn_hg_test_median_filter_1.txt")
predict_median_filter_3 = np.loadtxt("result/cnn_hg_test_median_filter_3.txt")
predict_median_filter_5 = np.loadtxt("result/cnn_hg_test_median_filter_5.txt")
predict_median_filter_7 = np.loadtxt("result/cnn_hg_test_median_filter_7.txt")
predict_median_filter_9 = np.loadtxt("result/cnn_hg_test_median_filter_9.txt")
predict_median_filter_11 = np.loadtxt("result/cnn_hg_test_median_filter_11.txt")
predict_median_filter_13 = np.loadtxt("result/cnn_hg_test_median_filter_13.txt")
predict_median_filter_15 = np.loadtxt("result/cnn_hg_test_median_filter_15.txt")
predict_median_filter_17 = np.loadtxt("result/cnn_hg_test_median_filter_17.txt")

# 均方误差 MSE
plt.figure()
plt.plot(predict_median_filter_1[:, 0], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(predict_median_filter_3[:, 0], color='#7d1e3f', linewidth=1, label='median_filter_3')
#plt.plot(predict_median_filter_5[:, 0], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(predict_median_filter_7[:, 0], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(predict_median_filter_9[:, 0], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(predict_median_filter_11[:, 0], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(predict_median_filter_13[:, 0], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(predict_median_filter_15[:, 0], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(predict_median_filter_17[:, 0], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("test MSE")
plt.xlabel('epoch')
plt.ylabel('test MSE')
# 显示图例
plt.legend()

# 决定系数 R-squared
plt.figure()
plt.plot(predict_median_filter_1[:, 1], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(predict_median_filter_3[:, 1], color='#7d1e3f', linewidth=1, label='median_filter_3')
plt.plot(predict_median_filter_5[:, 1], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(predict_median_filter_7[:, 1], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(predict_median_filter_9[:, 1], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(predict_median_filter_11[:, 1], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(predict_median_filter_13[:, 1], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(predict_median_filter_15[:, 1], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(predict_median_filter_17[:, 1], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("test R-squared")
plt.xlabel('epoch')
plt.ylabel('test R-squared')
# 显示图例
plt.legend()



# 平均绝对误差 RMSE
plt.figure()
plt.plot(predict_median_filter_1[:, 2], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(predict_median_filter_3[:, 2], color='#7d1e3f', linewidth=1, label='median_filter_3')
plt.plot(predict_median_filter_5[:, 2], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(predict_median_filter_7[:, 2], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(predict_median_filter_9[:, 2], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(predict_median_filter_11[:, 2], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(predict_median_filter_13[:, 2], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(predict_median_filter_15[:, 2], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(predict_median_filter_17[:, 2], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("test MAE")
plt.xlabel('epoch')
plt.ylabel('test MAE')
# 显示图例
plt.legend()


# 均方根误差 MAE
plt.figure()
plt.plot(predict_median_filter_1[:, 3], color='#1f3a64', linewidth=1, label='median_filter_1')
plt.plot(predict_median_filter_3[:, 3], color='#7d1e3f', linewidth=1, label='median_filter_3')
plt.plot(predict_median_filter_5[:, 3], color='#6a4c8c', linewidth=1, label='median_filter_5')
plt.plot(predict_median_filter_7[:, 3], color='#2d6a4f', linewidth=1, label='median_filter_7')
plt.plot(predict_median_filter_9[:, 3], color='#4b5320', linewidth=1, label='median_filter_9')
plt.plot(predict_median_filter_11[:, 3], color='#505050', linewidth=1, label='median_filter_11')
plt.plot(predict_median_filter_13[:, 3], color='#e76f51', linewidth=1, label='median_filter_13')
plt.plot(predict_median_filter_15[:, 3], color='#d9b310', linewidth=1, label='median_filter_15')
plt.plot(predict_median_filter_17[:, 3], color='#7b3f00', linewidth=1, label='median_filter_17')
plt.title("test RMSE")
plt.xlabel('epoch')
plt.ylabel('test RMSE')
# 显示图例
plt.legend()


# ##############################################################
# 优化器
# ##############################################################

Adagrad = np.loadtxt("result/cnn_hg_optimizer_Adagrad.txt")
Adam = np.loadtxt("result/cnn_hg_optimizer_Adam.txt")
momentum = np.loadtxt("result/cnn_hg_optimizer_momentum.txt")
SGD = np.loadtxt("result/cnn_hg_optimizer_SGD.txt")

Adagrad_test = np.loadtxt("result/cnn_hg_test_optimizer_Adagrad.txt")
Adam_test = np.loadtxt("result/cnn_hg_test_optimizer_Adam.txt")
momentum_test = np.loadtxt("result/cnn_hg_test_optimizer_momentum.txt")
SGD_test = np.loadtxt("result/cnn_hg_test_optimizer_SGD.txt")

# ##############################################################
# 优化器 训练集
# ##############################################################
# 均方误差 MSE
plt.figure()
plt.plot(Adagrad[:, 0], color='#1f3a64', linewidth=3, label='Adagrad')
plt.plot(Adam[:, 0], color='#7d1e3f', linewidth=3, label='Adam')
plt.plot(momentum[:, 0], color='#6a4c8c', linewidth=3, label='momentum')
plt.plot(SGD[:, 0], color='#2d6a4f', linewidth=3, label='SGD')
plt.title("optimizer MSE")
plt.xlabel('epoch')
plt.ylabel('optimizer MSE')
# 显示图例
plt.legend()

# 决定系数 R-squared
plt.figure()
plt.plot(Adagrad[:, 1], color='#1f3a64', linewidth=3, label='Adagrad')
plt.plot(Adam[:, 1], color='#7d1e3f', linewidth=3, label='Adam')
plt.plot(momentum[:, 1], color='#6a4c8c', linewidth=3, label='momentum')
plt.plot(SGD[:, 1], color='#2d6a4f', linewidth=3, label='SGD')
plt.title("optimizer R-squared")
plt.xlabel('epoch')
plt.ylabel('optimizer R-squared')
# 显示图例
plt.legend()



# 平均绝对误差 RMSE
plt.figure()
plt.plot(Adagrad[:, 2], color='#1f3a64', linewidth=3, label='Adagrad')
plt.plot(Adam[:, 2], color='#7d1e3f', linewidth=3, label='Adam')
plt.plot(momentum[:, 2], color='#6a4c8c', linewidth=3, label='momentum')
plt.plot(SGD[:, 2], color='#2d6a4f', linewidth=3, label='SGD')
plt.title("optimizer RMSE")
plt.xlabel('epoch')
plt.ylabel('optimizer RMSE')
# 显示图例
plt.legend()

# ##############################################################
# 优化器 测试集
# ##############################################################
# 均方误差 MSE
plt.figure()
plt.plot(Adagrad_test[:, 0], color='#1f3a64', linewidth=3, label='Adagrad')
plt.plot(Adam_test[:, 0], color='#7d1e3f', linewidth=3, label='Adam')
plt.plot(momentum_test[:, 0], color='#6a4c8c', linewidth=3, label='momentum')
plt.plot(SGD_test[:, 0], color='#2d6a4f', linewidth=3, label='SGD')
plt.title("optimizer MSE test")
plt.xlabel('epoch')
plt.ylabel('optimizer MSE test')
# 显示图例
plt.legend()

# 决定系数 R-squared
plt.figure()
plt.plot(Adagrad_test[:, 1], color='#1f3a64', linewidth=3, label='Adagrad')
plt.plot(Adam_test[:, 1], color='#7d1e3f', linewidth=3, label='Adam')
plt.plot(momentum_test[:, 1], color='#6a4c8c', linewidth=3, label='momentum')
plt.plot(SGD_test[:, 1], color='#2d6a4f', linewidth=3, label='SGD')
plt.title("optimizer R-squared test")
plt.xlabel('epoch')
plt.ylabel('optimizer R-squared test')
# 显示图例
plt.legend()

# 平均绝对误差 RMSE
plt.figure()
plt.plot(Adagrad_test[:, 2], color='#1f3a64', linewidth=3, label='Adagrad')
plt.plot(Adam_test[:, 2], color='#7d1e3f', linewidth=3, label='Adam')
plt.plot(momentum_test[:, 2], color='#6a4c8c', linewidth=3, label='momentum')
plt.plot(SGD_test[:, 2], color='#2d6a4f', linewidth=3, label='SGD')
plt.title("optimizer RMSE test")
plt.xlabel('epoch')
plt.ylabel('optimizer RMSE test')
# 显示图例
plt.legend()

"""


# ##############################################################
# learning rate
# ##############################################################
lr_0 = np.loadtxt("result/cnn_hg_lr_0.txt")
lr_1 = np.loadtxt("result/cnn_hg_lr_1.txt")
lr_2 = np.loadtxt("result/cnn_hg_lr_2.txt")
lr_3 = np.loadtxt("result/cnn_hg_lr_3.txt")
lr_4 = np.loadtxt("result/cnn_hg_lr_4.txt")
lr_5 = np.loadtxt("result/cnn_hg_lr_5.txt")
lr_6 = np.loadtxt("result/cnn_hg_lr_6.txt")
lr_7 = np.loadtxt("result/cnn_hg_lr_7.txt")
lr_8 = np.loadtxt("result/cnn_hg_lr_8.txt")
lr_9 = np.loadtxt("result/cnn_hg_lr_9.txt")
lr_10 = np.loadtxt("result/cnn_hg_lr_10.txt")
lr_20 = np.loadtxt("result/cnn_hg_lr_20.txt")
lr_30 = np.loadtxt("result/cnn_hg_lr_30.txt")

test_lr_0 = np.loadtxt("result/cnn_hg_test_lr_0.txt")
test_lr_1 = np.loadtxt("result/cnn_hg_test_lr_1.txt")
test_lr_2 = np.loadtxt("result/cnn_hg_test_lr_2.txt")
test_lr_3 = np.loadtxt("result/cnn_hg_test_lr_3.txt")
test_lr_4 = np.loadtxt("result/cnn_hg_test_lr_4.txt")
test_lr_5 = np.loadtxt("result/cnn_hg_test_lr_5.txt")
test_lr_6 = np.loadtxt("result/cnn_hg_test_lr_6.txt")
test_lr_7 = np.loadtxt("result/cnn_hg_test_lr_7.txt")
test_lr_8 = np.loadtxt("result/cnn_hg_test_lr_8.txt")
test_lr_9 = np.loadtxt("result/cnn_hg_test_lr_9.txt")
test_lr_10 = np.loadtxt("result/cnn_hg_test_lr_10.txt")
test_lr_20 = np.loadtxt("result/cnn_hg_test_lr_20.txt")
test_lr_30 = np.loadtxt("result/cnn_hg_test_lr_30.txt")

"""
# ##############################################################
# learning rate
# ##############################################################
# 均方误差 MSE
plt.figure()
plt.plot(lr_0[:, 0], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(lr_1[:, 0], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(lr_2[:, 0], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(lr_3[:, 0], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(lr_4[:, 0], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(lr_5[:, 0], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(lr_6[:, 0], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(lr_7[:, 0], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(lr_8[:, 0], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(lr_9[:, 0], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(lr_10[:, 0], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(lr_20[:, 0], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(lr_30[:, 0], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate train MSE")
plt.xlabel('epoch')
plt.ylabel('learning rate train MSE')
# 显示图例
plt.legend()

# 决定系数 R-squared
plt.figure()
plt.plot(lr_0[:, 1], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(lr_1[:, 1], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(lr_2[:, 1], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(lr_3[:, 1], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(lr_4[:, 1], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(lr_5[:, 1], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(lr_6[:, 1], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(lr_7[:, 1], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(lr_8[:, 1], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(lr_9[:, 1], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(lr_10[:, 1], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(lr_20[:, 1], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(lr_30[:, 1], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate train R-squared")
plt.xlabel('epoch')
plt.ylabel('learning rate train R-squared')
# 显示图例
plt.legend()

# 平均绝对误差 RMSE
plt.figure()
plt.plot(lr_0[:, 2], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(lr_1[:, 2], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(lr_2[:, 2], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(lr_3[:, 2], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(lr_4[:, 2], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(lr_5[:, 2], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(lr_6[:, 2], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(lr_7[:, 2], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(lr_8[:, 2], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(lr_9[:, 2], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(lr_10[:, 2], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(lr_20[:, 2], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(lr_30[:, 2], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate train RMSE")
plt.xlabel('epoch')
plt.ylabel('learning rate train RMSE')
# 显示图例
plt.legend()


# 均方根误差 MAE
plt.figure()
plt.plot(lr_0[:, 3], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(lr_1[:, 3], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(lr_2[:, 3], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(lr_3[:, 3], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(lr_4[:, 3], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(lr_5[:, 3], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(lr_6[:, 3], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(lr_7[:, 3], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(lr_8[:, 3], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(lr_9[:, 3], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(lr_10[:, 3], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(lr_20[:, 3], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(lr_30[:, 3], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate train MAE")
plt.xlabel('epoch')
plt.ylabel('learning rate train MAE')
# 显示图例
plt.legend()

# ##############################################################
# learning rate   test_
# ##############################################################


# 均方误差 MSE
plt.figure()
plt.plot(test_lr_0[:, 0], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(test_lr_1[:, 0], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(test_lr_2[:, 0], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(test_lr_3[:, 0], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(test_lr_4[:, 0], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(test_lr_5[:, 0], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(test_lr_6[:, 0], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(test_lr_7[:, 0], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(test_lr_8[:, 0], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(test_lr_9[:, 0], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(test_lr_10[:, 0], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(test_lr_20[:, 0], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(test_lr_30[:, 0], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate test MSE")
plt.xlabel('epoch')
plt.ylabel('learning rate test MSE')
# 显示图例
plt.legend()

# 决定系数 R-squared
plt.figure()
plt.plot(test_lr_0[:, 1], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(test_lr_1[:, 1], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(test_lr_2[:, 1], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(test_lr_3[:, 1], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(test_lr_4[:, 1], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(test_lr_5[:, 1], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(test_lr_6[:, 1], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(test_lr_7[:, 1], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(test_lr_8[:, 1], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(test_lr_9[:, 1], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(test_lr_10[:, 1], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(test_lr_20[:, 1], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(test_lr_30[:, 1], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate test R-squared")
plt.xlabel('epoch')
plt.ylabel('learning rate test R-squared')
# 显示图例
plt.legend()



# 平均绝对误差 RMSE
plt.figure()
plt.plot(test_lr_0[:, 2], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(test_lr_1[:, 2], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(test_lr_2[:, 2], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(test_lr_3[:, 2], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(test_lr_4[:, 2], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(test_lr_5[:, 2], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(test_lr_6[:, 2], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(test_lr_7[:, 2], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(test_lr_8[:, 2], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(test_lr_9[:, 2], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(test_lr_10[:, 2], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(test_lr_20[:, 2], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(test_lr_30[:, 2], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate test RMSE")
plt.xlabel('epoch')
plt.ylabel('learning rate test RMSE')
# 显示图例
plt.legend()


# 均方根误差 MAE
plt.figure()
plt.plot(test_lr_0[:, 3], color='#1f3a64', linewidth=1, label='lr = 0.0001')
plt.plot(test_lr_1[:, 3], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(test_lr_2[:, 3], color='#6a4c8c', linewidth=1, label='lr = 0.002')
plt.plot(test_lr_3[:, 3], color='#2d6a4f', linewidth=1, label='lr = 0.003')
plt.plot(test_lr_4[:, 3], color='#4b5320', linewidth=1, label='lr = 0.004')
plt.plot(test_lr_5[:, 3], color='#505050', linewidth=1, label='lr = 0.005')
plt.plot(test_lr_6[:, 3], color='#e76f51', linewidth=1, label='lr = 0.006')
plt.plot(test_lr_7[:, 3], color='#d9b310', linewidth=1, label='lr = 0.007')
plt.plot(test_lr_8[:, 3], color='#7b3f00', linewidth=1, label='lr = 0.008')
plt.plot(test_lr_9[:, 3], color='#2a7f62', linewidth=1, label='lr = 0.009')
plt.plot(test_lr_10[:, 3], color='#7a7e89', linewidth=1, label='lr = 0.01')
plt.plot(test_lr_20[:, 3], color='#9b2c53', linewidth=1, label='lr = 0.02')
plt.plot(test_lr_30[:, 3], color='#4e5b63', linewidth=1, label='lr = 0.03')
plt.title("learning rate test MAE")
plt.xlabel('epoch')
plt.ylabel('learning rate test MAE')
# 显示图例
plt.legend()

"""


plt.figure()
plt.plot(test_lr_1[:, 0], color='#7d1e3f', linewidth=1, label='lr = 0.001')
plt.plot(lr_1[:, 0], color='#4e5b63', linewidth=1, label='lr = 0.001')
plt.title("learning rate MSE")
plt.xlabel('epoch')
plt.ylabel('learning rate MSE')
# 显示图例
plt.legend()

"""
# ##############################################################
# 绘制三维散点
# ##############################################################


plt.figure()
plt.plot(Adam[:, 0], color='#7d1e3f', linewidth=3, label='train')
plt.plot(Adam_test[:, 0], color='#4e5b63', linewidth=3, label='test')
plt.title("MAE")
plt.xlabel('epoch')
plt.ylabel('MAE')
# 显示图例
plt.legend()
"""


plt.show()

