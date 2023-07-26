# -*- coding: utf-8 -*-
#
# Title: 提取脑电图特征
# Author: 计科2001-韩明辰
# Description: 使用TE RTE TSE RTSE提取脑电图特征
# Refer:
# Date: 2023-07-06
#


import os
import h5py
import mne
from mne.preprocessing import ICA
import numpy as np
import Entropy_dll
import matplotlib.pyplot as plt


#
# 读取数据
#
def read_evoked(h5path):
    evename_list = []
    evoked_list = []
    with h5py.File(h5path, "r") as f:
        for key in f.keys():
            evename_list.append(key)
            evoked_list.append(np.array(f[key]['value']))
    return evename_list, evoked_list


'''
fp1 输出文件
evoked_item 数据
delay 时延
k l 维度
name

'''
'''
进行特征提取
'''


def matrix_get(fp1, evoked_item, delay, k, l, name, div):
    print(name)
    print(name, file=fp1)
    matrix_len = evoked_item.shape[0]
    matrix = np.zeros((matrix_len, matrix_len))
    for i in range(0, matrix_len):
        print(i, end=' ')
        for j in range(0, matrix_len):
            evoked_item_i = Entropy_dll.Entropy.calculate_average(evoked_item[i], div)
            evoked_item_j = Entropy_dll.Entropy.calculate_average(evoked_item[j], div)
            evoked_item_te = Entropy_dll.Entropy(evoked_item_i, evoked_item_j)
            te = evoked_item_te.TE(delay, k, l)
            print(te, file=fp1, end=',')
            matrix[i][j] = te

        print('', file=fp1)

    print(matrix)


def matrix_get_iterval(fp1, evoked_item, delay, k, l, name, start, end, step):
    print(name)
    matrix_len = evoked_item.shape[0]
    print(len(evoked_item[0]))

    for opi in range(start, end, step):
        print(name, file=fp1)
        for i in range(0, matrix_len):
            for j in range(0, matrix_len):
                evoked_item_i = evoked_item[i][opi:opi + step]
                evoked_item_j = evoked_item[j][opi:opi + step]
                evoked_item_te = Entropy_dll.Entropy(evoked_item_i, evoked_item_j)
                te = evoked_item_te.TE(delay, k, l)
                print(te, file=fp1, end=',')
            print('', file=fp1)


def TSE_matrix_get_iterval(fp1, evoked_item, delay, k, l, name, start, end, step, alp):
    print(name)
    matrix_len = evoked_item.shape[0]
    print(len(evoked_item[0]))

    for opi in range(start, end, step):
        print(name, file=fp1)
        for i in range(0, matrix_len):
            for j in range(0, matrix_len):
                evoked_item_i = evoked_item[i][opi:opi + step]
                evoked_item_j = evoked_item[j][opi:opi + step]
                evoked_item_te = Entropy_dll.Entropy(evoked_item_i, evoked_item_j)
                if alp == 1.0:
                    tse = evoked_item_te.TSE(delay, k, l)
                else:
                    tse = evoked_item_te.RTSE(delay, k, l, alp)
                print(format(tse[7], '.17f'), file=fp1, end=',')
            print('', file=fp1)


def RSE_matrix_get_iterval(fp1, evoked_item, delay, k, l, name, start, end, step, alp):
    print(name)
    matrix_len = evoked_item.shape[0]
    print(len(evoked_item[0]))

    for opi in range(start, end, step):
        print(name, file=fp1)
        for i in range(0, matrix_len):
            for j in range(0, matrix_len):
                evoked_item_i = evoked_item[i][opi:opi + step]
                evoked_item_j = evoked_item[j][opi:opi + step]
                evoked_item_te = Entropy_dll.Entropy(evoked_item_i, evoked_item_j)
                rte = evoked_item_te.RTE(delay, k, l, alp)
                print(rte, file=fp1, end=',')
            print('', file=fp1)


def TE_create_data_txt(path, eegid):
    textpath = os.path.join(path, eegid)
    if os.path.isfile(textpath):
        print('ok!')
        return

    eegid_h5 = eegid + '.h5'
    h5path = os.path.join(r'./evoked', eegid_h5)

    if os.path.isfile(h5path) == False:
        print('NOT EXIT!')
        return

    evename_list, evoked_list = read_evoked(h5path)

    fp1 = open(textpath, 'w', encoding='utf-8')
    for i in range(0, len(evoked_list)):
        # matrix_get(fp1, evoked_list[i], 1, 2, 2, evename_list[i], 100)
        matrix_get_iterval(fp1, evoked_list[i], 1, 1, 1, evename_list[i], 5000, 8000, 50)


def TSE_create_data_txt(path, eegid):
    textpath = os.path.join(path, eegid)
    if os.path.isfile(textpath):
        print('ok!')
        return

    eegid_h5 = eegid + '.h5'
    h5path = os.path.join(r'./evoked', eegid_h5)

    if os.path.isfile(h5path) == False:
        print('NOT EXIT!')
        return

    evename_list, evoked_list = read_evoked(h5path)
    fp1 = open(textpath, 'w', encoding='utf-8')
    for i in range(0, len(evoked_list)):
        # matrix_get(fp1, evoked_list[i], 1, 2, 2, evename_list[i], 100)
        TSE_matrix_get_iterval(fp1, evoked_list[i], 1, 1, 1, evename_list[i], 0, 5000, 100, 1.0)
    fp1.close()
    del evoked_list
    del evename_list


def RTE_create_data_txt(path, eegid):
    eegid_h5 = eegid + '.h5'
    h5path = os.path.join(r'./evoked', eegid_h5)

    if os.path.isfile(h5path) == False:
        print('NOT EXIT!')
        return

    evename_list, evoked_list = read_evoked(h5path)
    for j in range(11, 21):
        alp = j / 10.0
        eegid2 = eegid + str(alp)
        textpath = os.path.join(path, eegid2)
        if os.path.isfile(textpath):
            print('ok!')
            continue
        fp1 = open(textpath, 'w', encoding='utf-8')
        for i in range(0, len(evoked_list)):
            RSE_matrix_get_iterval(fp1, evoked_list[i], 1, 1, 1, evename_list[i], 0, 5000, 50, alp)


def RTSE_create_data_txt(path, eegid):
    eegid_h5 = eegid + '.h5'
    h5path = os.path.join(r'./evoked', eegid_h5)

    if os.path.isfile(h5path) == False:
        print('NOT EXIT!')
        return

    evename_list, evoked_list = read_evoked(h5path)
    for j in range(10, 21):
        alp = j / 10.0
        eegid2 = eegid + str(alp)
        textpath = os.path.join(path, eegid2)
        if os.path.isfile(textpath):
            print('ok!')
            continue
        fp1 = open(textpath, 'w', encoding='utf-8')
        for i in range(0, len(evoked_list)):
            TSE_matrix_get_iterval(fp1, evoked_list[i], 1, 1, 1, evename_list[i], 0, 1000, 100, alp)


def image(x, y):
    fig, ax = plt.subplots()  # 创建图实例
    ax.plot(x, label='X')  # 作y1 = x 图，并标记此线名为linear
    ax.plot(y, label='Y')  # 作y2 = x^2 图，并标记此线名为quadratic
    ax.set_xlabel('x label')  # 设置x轴名称 x label
    ax.set_ylabel('y label')  # 设置y轴名称 y label
    ax.set_title('Simple Plot')  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.show()  # 图形可视化


def show(eegid1, eegid2, num):
    eegid_h5_1 = eegid1 + '.h5'
    eegid_h5_2 = eegid2 + '.h5'
    h5path1 = os.path.join(r'./evoked', eegid_h5_1)
    h5path2 = os.path.join(r'./evoked', eegid_h5_2)
    if os.path.isfile(h5path1) == False:
        print('NOT EXIT!')
        return
    if os.path.isfile(h5path2) == False:
        print('NOT EXIT!')
        return

    evename_list1, evoked_list1 = read_evoked(h5path1)
    evename_list2, evoked_list2 = read_evoked(h5path2)
    image(evoked_list1[0][0][0:num], evoked_list1[1][0][0:num])
