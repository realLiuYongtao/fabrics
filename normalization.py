# 数据进行归一化
import numpy as np
import matplotlib.pyplot as plt

d_num = 232
j_num = 40
m_num = 128
h_num = 208

# 定义裁剪的范围
x_upper = 1600
x_lower = 1309


def Min_Max_Normalization():
    plt.figure()
    for i in range(1, d_num + 1):
        txt = "d/d_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)
        plt.plot(x, yy, color=[i / (d_num * 1.5), 0.3, (d_num - i) / (d_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(d)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.figure()
    for i in range(1, m_num + 1):
        txt = "m/m_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)
        plt.plot(x, yy, color=[i / (m_num * 1.5), 0.3, (m_num - i) / (m_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(m)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.figure()
    for i in range(1, j_num + 1):
        txt = "j/j_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)
        plt.plot(x, yy, color=[i / (j_num * 1.5), 0.3, (j_num - i) / (j_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(j)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.figure()
    for i in range(1, h_num + 1):
        txt = "h/h_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)
        plt.plot(x, yy, color=[i / (h_num * 1.5), 0.3, (h_num - i) / (h_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(h)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.show()

    return


def Min_Max_Normalization_save_data():
    plt.figure()
    for i in range(1, d_num + 1):
        txt = "d/d_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)

        # 将 xx 和 yy 保存到文件
        output_txt = "normalization_data/d/d_{}.txt".format(i)
        np.savetxt(output_txt, np.column_stack((yy, x)), fmt="%.8f", comments='')

        plt.plot(x, yy, color=[i / (d_num * 1.5), 0.3, (d_num - i) / (d_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(d)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.figure()
    for i in range(1, m_num + 1):
        txt = "m/m_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)

        # 将 xx 和 yy 保存到文件
        output_txt = "normalization_data/m/m_{}.txt".format(i)
        np.savetxt(output_txt, np.column_stack((yy, x)), fmt="%.8f", comments='')

        plt.plot(x, yy, color=[i / (m_num * 1.5), 0.3, (m_num - i) / (m_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(m)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.figure()
    for i in range(1, j_num + 1):
        txt = "j/j_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)

        # 将 xx 和 yy 保存到文件
        output_txt = "normalization_data/j/j_{}.txt".format(i)
        np.savetxt(output_txt, np.column_stack((yy, x)), fmt="%.8f", comments='')

        plt.plot(x, yy, color=[i / (j_num * 1.5), 0.3, (j_num - i) / (j_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(j)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.figure()
    for i in range(1, h_num + 1):
        txt = "h/h_{}.txt".format(i)
        data = np.loadtxt(txt)
        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 筛选 y 值在 x_upper 到 x_lower 之间的数据
        mask = (x >= x_lower) & (x <= x_upper)
        x = x[mask]
        y = y[mask]

        Max = max(y)
        Min = min(y)
        yy = (y - Min) / (Max - Min)

        # 将 xx 和 yy 保存到文件
        output_txt = "normalization_data/h/h_{}.txt".format(i)
        np.savetxt(output_txt, np.column_stack((yy, x)), fmt="%.8f", comments='')

        plt.plot(x, yy, color=[i / (h_num * 1.5), 0.3, (h_num - i) / (h_num * 1.5)], linewidth=1)
        plt.title('Min_Max_Normalization(h)')
        plt.xlabel('wavelength(nm)')
        plt.ylabel('y')

    plt.show()

    return


if __name__ == "__main__":
    # Min_Max_Normalization()
    Min_Max_Normalization_save_data()
