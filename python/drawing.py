# -*- coding: utf-8 -*-
#
# Title: 画图
# Author: 计科2001-韩明辰
# Description:
# Refer:
# Date: 2023-07-15
#

import os
import h5py
import mne
from mne.preprocessing import ICA
import numpy as np
import Entropy_dll
import matplotlib.pyplot as plt
from matplotlib import ticker, patches
from mpl_toolkits.mplot3d import Axes3D
import random
import pylab as mpl
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import classification_TE

'''
读入数据
'''


def read_TE():
    # 读取数据文件
    with open(r'.\dataset\TE\result\静息_语义', 'r') as file:
        # 读取第一行数据并转换为列表
        line1 = file.readline().strip().split()
        # 读取第二行数据并转换为列表
        line2 = file.readline().strip().split()

    # 转换数据类型为数值
    line1 = list(map(float, line1))
    line2 = list(map(float, line2))

    line3 = []
    for i in range(0, len(line1)):
        line3.append(line1[i] / (line1[i] + line2[i]))
    file.close()
    with open(r'.\dataset\TE\result\静息_语音', 'r') as file1:
        # 读取第一行数据并转换为列表
        line1_1 = file1.readline().strip().split()
        # 读取第二行数据并转换为列表
        line2_1 = file1.readline().strip().split()

    # 转换数据类型为数值
    line1_1 = list(map(float, line1_1))
    line2_1 = list(map(float, line2_1))
    line3_1 = []
    for i in range(0, len(line1_1)):
        line3_1.append(line1_1[i] / (line1_1[i] + line2_1[i]))

    # 开始画图
    colors = [
        'lightcoral',
        'darkorange',
        'gold',
        'palegreen',
        'paleturquoise',
        'skyblue',
        'plum',
        'hotpink',
        'pink']

    plt.title('Result Analysis')
    plt.figure(figsize=(10, 8), dpi=250)
    # figure configurations
    plt.rcParams.update({"font.size": 16})
    plt.ylim(0.0, 1.1)

    # set y axis a format of percentage
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

    # 时间轴
    time = np.arange(0, len(line1), 1)
    # 绘制折线图
    plt.plot(time, line3_1, label='Speech Fluency', color=colors[0])
    plt.plot(time, line3, label='Semantic Fluency', color=colors[2])

    # 设置图例和标题
    plt.legend()

    # 显示图形
    plt.show()


def read_TSE(num, file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines_num = num * 4 + 1
    # 存储所有任务的数据
    tasks_data = []
    task_data = {}

    # 迭代每一行
    for i, line in enumerate(lines):
        line = line.strip()  # 去除行末尾的换行符和空格
        # 如果行号为奇数，则获取测试名称
        if i % lines_num == 0:
            # 如果不是第一个任务，则将上一个任务的数据添加到任务列表中
            if task_data:
                tasks_data.append(task_data)

            # 重置当前任务的数据字典
            task_data = {}

            # 获取测试名称
            test_name = line
            task_data["test_name"] = test_name

        elif (i % lines_num) % 4 == 1:
            result_label = line.split()[0]  # 获取结果标号
            precision = float((lines[i + 1].split(":"))[1])  # 获取精确度
            recall = float((lines[i + 2].split(":"))[1])  # 获取召回率
            f1 = float((lines[i + 3].split(":"))[1])  # 获取F1值

            # 将结果存储在任务数据字典中
            if "results" not in task_data:
                task_data["results"] = {}

            task_data["results"][result_label] = {
                "precision": precision,
                "recall": recall,
                "F1": f1
            }

    # 将最后一个任务的数据添加到任务列表中
    if task_data:
        tasks_data.append(task_data)

    return tasks_data


def plt_3D():
    tasks_data = read_TSE()
    # 输出每个任务的测试名称和结果
    for i, task_data in enumerate(tasks_data):
        print(f"任务 {i + 1}:")
        print("测试名称：", task_data["test_name"])
        print("测试结果：", task_data["results"])
        print()

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    colors = [
        'lightcoral',
        'darkorange',
        'gold',
        'palegreen',
        'paleturquoise',
        'skyblue',
        'plum',
        'hotpink',
        'pink']
    plt.figure(figsize=(10, 8), dpi=150)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = []
    for i in range(10, 21):
        xs.append(i / 10.0)

    for i, task_data in enumerate(tasks_data):
        ys = []
        for j in task_data["results"].values():
            ys.append(j["F1"])
        ax.plot(xs, ys, zs=i, zdir='y', color=colors[i], marker='o', alpha=0.8)

    # 在设置zdir = 'y'的情形下，其实y轴才是z轴，然后z轴变成了y轴
    ax.set_xlabel(' order alpha ')
    ax.set_ylabel('task')
    ax.set_zlabel('F1')

    plt.show()


'''
画Alp值图
'''


def plt_RSE():
    path = r'.\dataset\RTE\result\compare'
    tasks_data = read_TSE(11, path)

    xs = []
    for i in range(10, 21):
        xs.append(i / 10.0)

    # 开始画图
    sub_axix = filter(lambda x: x % 200 == 0, xs)
    colors = [
        'lightcoral',
        'darkorange',
        'gold',
        'palegreen',
        'paleturquoise',
        'skyblue',
        'plum',
        'hotpink',
        'pink']

    plt.title('Result Analysis')
    plt.figure(figsize=(10, 8), dpi=250)

    for i, task_data in enumerate(tasks_data):
        ys = []
        for j in task_data["results"].values():
            ys.append(j["F1"])

        color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))  # 得到一个随机的颜色用于下面绘制该条折线图
        plt.plot(xs, ys, color=colors[i], label=task_data["test_name"])

    plt.legend(loc='center right')
    plt.xlabel("order alpha")
    plt.ylabel("F1")
    plt.savefig('./picture/test2.jpg')
    plt.show()


'''
化频率图
'''


def plt_TSE():
    path = r'.\dataset\TSE\result\compare'
    tasks_data = read_TSE(16, path)
    xs = []
    for i in range(1, 17):
        xs.append(i * 120.0 / 32)

        # 开始画图
    sub_axix = filter(lambda x: x % 200 == 0, xs)
    colors = [
        'lightcoral',
        'darkorange',
        'gold',
        'palegreen',
        'paleturquoise',
        'skyblue',
        'plum',
        'hotpink',
        'pink']

    plt.title('Result Analysis')
    plt.figure(figsize=(10, 8), dpi=250)

    for i, task_data in enumerate(tasks_data):
        ys = []
        for j in task_data["results"].values():
            ys.append(j["F1"])

        plt.plot(xs, ys, color=colors[i], label=task_data["test_name"])

    plt.legend(loc='center right')
    plt.xlabel("frequency")
    plt.ylabel("F1")
    plt.savefig('./picture/test2.jpg')
    plt.show()


def plt_RTSE():
    # 更改路径即可
    path = r'.\dataset\RTSE\result\Semantic Fluency'

    tasks_data = read_TSE(16, path)
    print(tasks_data)

    plt.figure(figsize=(10, 8), dpi=500)
    fig = plt.figure(figsize=plt.figaspect(0.6), dpi=200)
    ax = fig.add_subplot(projection='3d')

    # Make data.
    xs = []
    for i in range(1, 17):
        xs.append(i * 120.0 / 32)
    ys = np.arange(1.0, 2.1, 0.1)
    zs = np.zeros((len(ys), len(xs)))
    xs, ys = np.meshgrid(xs, ys)

    for i, task_data in enumerate(tasks_data):
        j = 0
        for data_task in task_data["results"].values():
            zs[i][j] = data_task["F1"]
            j = j + 1

    # Plot the surface.
    surf = ax.plot_surface(xs, ys, zs, cmap='rainbow', linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlabel('F1')
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('Alpha')
    ax.set_zlim(0, 1.00)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5, location='left')

    plt.savefig('./picture/2.jpg')
    plt.show()


'''
ROC画图
'''


def roc_analysis():
    '''
    调用函数来生成数据
    '''
    from cross_check import TE_cross_check
    # data = TE_cross_check()
    data = classification_TE.test1()
    '''
    测试
    '''

    print(data)
    # 示例数据
    thresholds_item = [entry[0] for entry in data]  # 获取阈值（数组元素的第一个概率）
    thresholds = set(thresholds_item)
    thresholds = sorted(thresholds)
    true_positive_rates = []
    false_positive_rates = []

    positive_count = sum(entry[1] == 1 for entry in data)  # 正样本数量
    negative_count = sum(entry[1] == 0 for entry in data)  # 负样本数量

    for threshold in thresholds:
        true_positives = sum(entry[0] >= threshold and entry[1] == 1 for entry in data)
        false_positives = sum(entry[0] >= threshold and entry[1] == 0 for entry in data)

        true_positive_rate = true_positives / positive_count
        false_positive_rate = false_positives / negative_count

        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    true_positive_rates.append(0)
    false_positive_rates.append(0)
    print(true_positive_rates, false_positive_rates)
    AUC = 0
    for i in range(0, len(true_positive_rates) - 1):
        AUC = AUC + (false_positive_rates[i] - false_positive_rates[i + 1]) * (
                true_positive_rates[i] + true_positive_rates[i + 1]) / 2
    # 绘制ROC曲线
    print(f"AUC:{AUC}")
    plt.figure(figsize=(10, 8), dpi=250)
    label_auc = 'Stroop(AUC=' + str(round(AUC, 2)) + ')'
    print(label_auc)
    plt.plot(false_positive_rates, true_positive_rates, label=label_auc)
    plt.legend(loc='upper left')
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制随机猜测线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.savefig('./picture/test2.jpg')
    plt.show()


def roc_task():
    from sklearn.metrics import roc_curve, auc
    '''
    调用函数来生成数据
    '''
    from cross_check import TE_cross_check
    from cross_check import TSE_cross_check
    #data = TE_cross_check()
    data = [[0.6868000000000003, 1], [0.3854000000000001, 0], [0.5915999999999999, 1], [0.3399999999999999, 0],
            [0.39660000000000006, 1], [0.33160000000000006, 0], [0.6894000000000002, 1], [0.26799999999999996, 0],
            [0.6224000000000001, 1], [0.3797999999999999, 0], [0.6871999999999998, 1], [0.3215999999999999, 0],
            [0.6608000000000002, 1], [0.4143999999999999, 0], [0.5711999999999998, 1], [0.3975999999999999, 0],
            [0.6348000000000001, 1], [0.28979999999999995, 0], [0.5948000000000001, 1], [0.5288, 0]]

    y_label = []
    y_pre = []
    for i in range(0, len(data)):
        y_label.append(data[i][1])
        y_pre.append(data[i][0])
    print(y_label, y_pre)

    fpr, tpr, thersholds = roc_curve(y_label, y_pre)

    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(fpr, tpr, label='ROC (area = {0:.3f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.plot([-0.05, 1.05], [-0.05, 1.05], 'k--')  # 绘制随机猜测线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./picture/test2.jpg')
    plt.show()

    max_threshold = 0
    max_precision = 0
    max_recall = 0
    max_F1 = 0

    for i, value in enumerate(thersholds):
        threshold = value
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for item in data:
            if item[0] >= threshold:
                if item[1] == 1:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                if item[1] == 1:
                    FN = FN + 1
                else:
                    TN = TN + 1

        if TP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * (precision * recall) / (precision + recall)
            if F1 > max_F1:
                max_threshold = threshold
                max_precision = precision
                max_recall = recall
                max_F1 = F1

    print(f"threshold:{max_threshold}")
    print(f"precision:{max_precision}")
    print(f"recall:{max_recall}")
    print(f"F1:{max_F1}")


'''
脑电极空间图
'''


def plot_electrode_map():
    chan_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
                 'Cz', 'Pz', 'A1', 'A2', 'M1', 'M2']
    all_chan = ['Cz', 'C4', 'T4', 'C3', 'T3', 'Fz', 'Fpz', 'Pz', 'T5', 'O1', 'O2', 'T6', 'F7', "Fp1", "Fp2", 'F8', 'F3',
                'F4', 'P3', "P4"]
    from Hypothesis_Testing import TE_Hypothesis_Test
    from Hypothesis_Testing import TSE_Hypothesis_Test
    matrices = TE_Hypothesis_Test()

    fp1 = open(r"./dataset/electrode_map/connectivity.txt", 'w', encoding='utf-8')
    for i in range(0, 23):
        for j in range(0, 23):
            if matrices[i][j] > 0.75:
                if chan_list[i] in all_chan and chan_list[j] in all_chan:
                    if matrices[i][j] == 3:
                        print(f"{chan_list[i]}, {chan_list[j]},red", file=fp1)
                    elif matrices[i][j] == 2:
                        print(f"{chan_list[i]}, {chan_list[j]},yellow", file=fp1)
                    elif matrices[i][j] == 1:
                        print(f"{chan_list[i]}, {chan_list[j]},blue", file=fp1)

    fp1.close()

    coordinate = {
        'Cz': (0, 0),
        'C4': (0.5, 0),
        'T4': (1, 0),
        'C3': (-0.5, 0),
        'T3': (-1, 0),
        'Fz': (0, 0.5),
        'Fpz': (0, 1),
        'Pz': (0, -0.5),
        'T5': (-np.cos(np.pi * 1 / 5), -np.sin(np.pi * 1 / 5)),
        'O1': (-np.cos(np.pi * 2 / 5), -np.sin(np.pi * 2 / 5)),
        'O2': (np.cos(np.pi * 2 / 5), -np.sin(np.pi * 2 / 5)),
        'T6': (np.cos(np.pi * 1 / 5), -np.sin(np.pi * 1 / 5)),
        'F7': (-np.cos(np.pi * 1 / 5), np.sin(np.pi * 1 / 5)),
        "Fp1": (-np.cos(np.pi * 2 / 5), np.sin(np.pi * 2 / 5)),
        "Fp2": (np.cos(np.pi * 2 / 5), np.sin(np.pi * 2 / 5)),
        'F8': (np.cos(np.pi * 1 / 5), np.sin(np.pi * 1 / 5)),
        'F3': (-0.45, 0.55),
        'F4': (0.45, 0.55),
        'P3': (-0.45, -0.55),
        "P4": (0.45, -0.55),
    }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=250)
    # plt.figure(figsize=(10, 8), dpi=250)
    # 画脸
    circle = patches.Circle(xy=(0, 0), radius=1.2, facecolor="w", edgecolor='k', lw=1, zorder=-2)
    ax.add_patch(circle)

    # 画耳朵
    ellipse1 = patches.Ellipse(xy=(-1.2, 0), width=0.5, height=1, facecolor="w", edgecolor='k', lw=1, zorder=-3)
    ax.add_patch(ellipse1)
    ellipse2 = patches.Ellipse(xy=(1.2, 0), width=0.5, height=1, facecolor="w", edgecolor='k', lw=1, zorder=-3)
    ax.add_patch(ellipse2)

    # 画鼻子
    angle = 1 / 2.5 * np.pi
    ax.plot([np.cos(angle) * 1.2, 0., -np.cos(angle) * 1.2], [np.sin(angle) * 1.2, 1.4, np.sin(angle) * 1.2], c='k',
            lw=1)

    # 画电极
    for x in coordinate.keys():
        ax.scatter(coordinate[x][0], coordinate[x][1], s=400, c='w', edgecolors='k')
        ax.text(coordinate[x][0], coordinate[x][1], s=x, ha="center", va="center", fontsize=10)

    # 画相关线条
    file = open(r"./dataset/electrode_map/connectivity.txt", 'r')
    for line in file:
        if len(line.split(",")) < 4:
            a1, a2, color = line.strip().split(",")
            x, y = zip(coordinate[a1.strip()], coordinate[a2.strip()])
            ax.plot(x, y, c=color.strip(), zorder=-1)
            print(line)

    # print(file)
    ax.axis("off")
    plt.savefig('./picture/test2.jpg')

    plt.show()


if __name__ == '__main__':
    '''
    画出ROC曲线 TE或TSE的 
    调用cross_check.py的TE_cross_check()和TSE_cross_check()函数
    '''
    roc_task()
    '''
    画出脑电图 
    调用Hypothesis_Testing.py的TE_Hypothesis_Test()函数
    '''
    # plot_electrode_map()
    '''
    画出RTSE的频率 Alp和F1值图 
    路径在函数内
    '''
    # plt_RTSE()
    '''
    画出TSE的频率和F1值图
    路径在函数内
    '''
    # plt_TSE()
    '''
    画出TE和Alp的图片
    路径在函数内
    '''
    # plt_RSE()
    '''
    画出F1值
    '''
    # F1_task()
