# 数据预处理
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.signal import savgol_filter

d_num = 232
j_num = 40
m_num = 128
h_num = 208

# 中值滤波
def median_filter(kernel_size):
    # 涤纶
    plt.figure()
    for i in range(1, d_num + 1):
        txt = "normalization_data/d/d_{}.txt".format(i)
        data = np.loadtxt(txt)

        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 对y数据进行中值滤波，窗口大小为3
        y_filtered = medfilt(y, kernel_size=kernel_size)

        # 将 xx 和 yy 保存到文件
        output_txt = "median_filter_data/{}/d/d_{}.txt".format(kernel_size,i)
        np.savetxt(output_txt, np.column_stack((y_filtered, x)), fmt="%.8f", comments='')

        plt.plot(x, y_filtered, color=[i / (d_num * 1.5), 0.3, (d_num - i) / (d_num * 1.5)], linewidth=1)
    plt.title('polyester(d)')
    plt.xlabel('wavelength(nm)')
    plt.ylabel('y')

    # 棉
    plt.figure()
    for i in range(1, m_num+1):
        txt = "normalization_data/m/m_{}.txt".format(i)
        data = np.loadtxt(txt)

        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 对y数据进行中值滤波
        y_filtered = medfilt(y, kernel_size=kernel_size)

        # 将 xx 和 yy 保存到文件
        output_txt = "median_filter_data/{}/m/m_{}.txt".format(kernel_size,i)
        np.savetxt(output_txt, np.column_stack((y_filtered, x)), fmt="%.8f", comments='')


        plt.plot(x, y_filtered, color=[i / (m_num * 1.5), 0.3, (m_num - i) / (m_num * 1.5)], linewidth=1)
    plt.title('cotton(m)')
    plt.xlabel('wavelength(nm)')
    plt.ylabel('y')

    # 锦纶
    plt.figure()
    for i in range(1, j_num+1):
        txt = "normalization_data/j/j_{}.txt".format(i)
        data = np.loadtxt(txt)

        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 对y数据进行中值滤波，窗口大小为3
        y_filtered = medfilt(y, kernel_size=kernel_size)

        # 将 xx 和 yy 保存到文件
        output_txt = "median_filter_data/{}/j/j_{}.txt".format(kernel_size,i)
        np.savetxt(output_txt, np.column_stack((y_filtered, x)), fmt="%.8f", comments='')

        plt.plot(x, y_filtered, color=[i / (j_num * 1.5), 0.3, (j_num - i) / (j_num * 1.5)], linewidth=1)
    plt.title('nylon(j)')
    plt.xlabel('wavelength(nm)')
    plt.ylabel('y')

    # 混纺
    plt.figure()
    for i in range(1, h_num+1):
        txt = "normalization_data/h/h_{}.txt".format(i)
        data = np.loadtxt(txt)

        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 对y数据进行中值滤波，窗口大小为3
        y_filtered = medfilt(y, kernel_size=kernel_size)

        # 将 xx 和 yy 保存到文件
        output_txt = "median_filter_data/{}/h/h_{}.txt".format(kernel_size,i)
        np.savetxt(output_txt, np.column_stack((y_filtered, x)), fmt="%.8f", comments='')

        plt.plot(x, y_filtered, color=[i / (h_num * 1.5), 0.3, (h_num - i) / (h_num * 1.5)], linewidth=1)
    plt.title('nylon(h)')
    plt.xlabel('wavelength(nm)')
    plt.ylabel('y')

    # 显示图表
    plt.show()


# sg平滑
def sg(window_length,polyorder):
    '''
    # 涤纶
    plt.figure()
    for i in range(1, d_num + 1):
        txt = "normalization_data/d/d_{}.txt".format(i)
        data = np.loadtxt(txt)

        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 使用Savitzky-Golay滤波器进行平滑，窗口大小为5，多项式阶数为2
        y_smoothed = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        # 将 xx 和 yy 保存到文件
        output_txt = "sg_data/{}_{}/d/d_{}.txt".format(window_length,polyorder, i)
        np.savetxt(output_txt, np.column_stack((y_smoothed, x)), fmt="%.8f", comments='')

        plt.plot(x, y_smoothed, color=[i / (d_num * 1.5), 0.3, (d_num - i) / (d_num * 1.5)], linewidth=1)
    plt.title('polyester(d)')
    plt.xlabel('wavelength(nm)')
    plt.ylabel('y')

    # 棉
    plt.figure()
    for i in range(1, m_num):
        txt = "normalization_data/m/m_{}.txt".format(i)
        data = np.loadtxt(txt)

        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 使用Savitzky-Golay滤波器进行平滑，窗口大小为5，多项式阶数为2
        y_smoothed = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        # 将 xx 和 yy 保存到文件
        output_txt = "sg_data/{}_{}/m/m_{}.txt".format(window_length,polyorder, i)
        np.savetxt(output_txt, np.column_stack((y_smoothed, x)), fmt="%.8f", comments='')

        plt.plot(x, y_smoothed, color=[i / (m_num * 1.5), 0.3, (m_num - i) / (m_num * 1.5)], linewidth=1)
    plt.title('cotton(m)')
    plt.xlabel('wavelength(nm)')
    plt.ylabel('y')
    '''
    # 锦纶
    #plt.figure()
    for i in range(1, j_num):
        txt = "normalization_data/j/j_{}.txt".format(i)
        data = np.loadtxt(txt)

        y = data[:, 0]  # data[:, 0]表示取所有行的第0列数据，即y坐标
        x = data[:, 1]  # data[:, 1]表示取所有行的第1列数据，即x坐标

        # 使用Savitzky-Golay滤波器进行平滑，窗口大小为5，多项式阶数为2
        y_smoothed = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        # 将 xx 和 yy 保存到文件
        output_txt = "sg_data/{}_{}/j/j_{}.txt".format(window_length,polyorder, i)
        np.savetxt(output_txt, np.column_stack((y_smoothed, x)), fmt="%.8f", comments='')

        plt.plot(x, y_smoothed, color=[i / (m_num * 1.5), 0.3, (m_num - i) / (m_num * 1.5)], linewidth=1)
    plt.title('nylon(j)')
    plt.xlabel('wavelength(nm)')
    plt.ylabel('y')

    # 显示图表
    plt.show()


if __name__ == "__main__":

    median_filter(11)
    median_filter(13)
    median_filter(15)
    median_filter(17)
    # sg(5,2)

    print()
