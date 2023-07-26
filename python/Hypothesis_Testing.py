# -*- coding: utf-8 -*-

#
# Title:实现假设检验
# Author: 计科2001-韩明辰
# Description: 使用机器学习进行假设检验
# Refer:
# Date: 2023-07-17
#


import os
import h5py
import mne
from mne.preprocessing import ICA
import numpy as np
import Entropy_dll
import matplotlib.pyplot as plt

'''
TE假设检验
使用了classification_TE.py的函数
数据path = r'./dataset/TE/dataset_50/'

'''


def TE_Hypothesis_Test():
    from statsmodels.stats import weightstats

    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U']
    from classification_TE import train

    hypothesis_matrice = np.zeros((23, 23))

    for eegid in all_eegid:

        matrices = []
        labels = []
        '''
        数据路径
        '''
        path = r'./dataset/TE/dataset_50/'
        # 改变通道
        index = ['A1+A2 OFF', '6.30']

        matrices, labels = train(path, eegid, matrices, labels, index)
        # 假设检验的示例数据（两组样本）
        for i in range(0, 23):
            for j in range(0, 23):
                if i == j:
                    continue
                group1 = []
                group2 = []
                for k in range(0, len(matrices)):
                    if labels[k] == index[0][0]:
                        group1.append(matrices[k][i][j])
                    else:
                        group2.append(matrices[k][i][j])
                t_stat, p_value, df = weightstats.ttest_ind(group1, group2)
                hypothesis_matrice[i][j] = hypothesis_matrice[i][j] + p_value
    hypothesis_matrice = hypothesis_matrice / len(all_eegid)
    for i in range(0, 23):
        for j in range(0, 23):
            if i == j:
                hypothesis_matrice[i][j] = 0
                continue
            if hypothesis_matrice[i][j] < 0.001:
                hypothesis_matrice[i][j] = 3
            elif hypothesis_matrice[i][j] < 0.01:
                hypothesis_matrice[i][j] = 2
            elif hypothesis_matrice[i][j] < 0.05:
                hypothesis_matrice[i][j] = 1
            else:
                hypothesis_matrice[i][j] = 0

    return hypothesis_matrice


'''
TSE假设检验
使用了classification_TE.py的函数
数据path = r'./dataset/TSE/dataset_50_test/'
'''


def TSE_Hypothesis_Test():
    from statsmodels.stats import weightstats

    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U']
    from classification_TE import train

    hypothesis_matrice = np.zeros((23, 23))

    for eegid in all_eegid:

        matrices = []
        labels = []
        '''
        数据路径
        '''
        path = r'./dataset/TSE/dataset_50_test/'
        # 改变通道
        index = ['A1+A2 OFF', '1.10']

        matrices, labels = train(path, eegid, matrices, labels, index)
        # 假设检验的示例数据（两组样本）
        for i in range(0, 23):
            for j in range(0, 23):
                if i == j:
                    continue
                group1 = []
                group2 = []
                for k in range(0, len(matrices)):
                    if labels[k] == index[0][0]:
                        group1.append(matrices[k][i][j])
                    else:
                        group2.append(matrices[k][i][j])
                t_stat, p_value, df = weightstats.ttest_ind(group1, group2)
                hypothesis_matrice[i][j] = hypothesis_matrice[i][j] + p_value
    hypothesis_matrice = hypothesis_matrice / len(all_eegid)
    for i in range(0, 23):
        for j in range(0, 23):
            if i == j:
                hypothesis_matrice[i][j] = 0
                continue
            if hypothesis_matrice[i][j] < 0.001:
                hypothesis_matrice[i][j] = 3
            elif hypothesis_matrice[i][j] < 0.01:
                hypothesis_matrice[i][j] = 2
            elif hypothesis_matrice[i][j] < 0.05:
                hypothesis_matrice[i][j] = 1
            else:
                hypothesis_matrice[i][j] = 0

    return hypothesis_matrice
