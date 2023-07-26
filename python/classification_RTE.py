# -*- coding: utf-8 -*-
#
# Title: 使用决策树进行分析
# Author: 计科2001-韩明辰
# Description: 使用决策树进行分析
# Refer:
# Date: 2023-07-17
#

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

'''
TSE Alp从1.0-2.0的F1值
使用classification_TE的函数
'''


def test1():
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U', 'MA00100V', 'MA00100W', 'MA00100X',
                 'MA00100Y', 'MA00100Z', 'MA001010', 'MA001011', 'MA001012', 'MA001013']
    '''
    训练集和测试集
    '''
    train_eegid = ['MA00100A']
    exam_eegid = ['MA00100B']
    matrices = []
    labels = []
    '''
    数据路径
    '''
    path = r'./dataset/RTE/dataset_50/'
    path1 = r'./dataset/RTE/dataset_50/'
    '''
    训练随机森林
    '''

    import classification_TE
    for i in range(10, 21):
        alp = i / 10.0
        index = ['A1+A2 OFF', '1.10', '1.20', '1.30']
        for eegid in train_eegid:
            eegid2 = eegid + str(alp)
            matrices, labels = classification_TE.train(path, eegid2, matrices, labels, index)

        classifier = classification_TE.random_forest_classification(matrices, labels)
        '''
        测试随机森林
        '''
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        probability_all = []
        probability_aLL_unit = []
        for eegid in exam_eegid:
            eegid2 = eegid + str(alp)
            TP_flag, FP_flag, FN_flag, TN_flag, probability_all, probability_aLL_unit = classification_TE.exam_confusion(
                path, eegid2, classifier, index)
            TP = TP + TP_flag
            FP = FP + FP_flag
            FN = FN + FN_flag
            TN = TN + TN_flag

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)
        print(alp)
        print(f"precision:{precision}")
        print(f"recall:{recall}")
        print(f"F1:{F1}")


if __name__ == '__main__':
    '''
    计算TSE的值
    '''
    test1()
