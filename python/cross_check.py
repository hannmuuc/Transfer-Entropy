# -*- coding: utf-8 -*-
#
# Title: 交叉检验
# Author: 计科2001-韩明辰
# Description: 对机器学习进行交叉检验
# Refer:
# Date: 2023-07-15
#

from sklearn.model_selection import KFold
import numpy as np

'''

调用classification_TE.py的test2()进行分类
'''


def TE_cross_check():
    eegid_all = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U']

    X = np.arange(42).reshape(21, 2)
    probability_tannel = []

    kf = KFold(n_splits=10, shuffle=True)  # 初始化KFold
    for train_index, test_index in kf.split(X):  # 调用split方法切分数据
        print('train_index:%s , test_index: %s ' % (train_index, test_index))
        train_eegid = []
        exam_eegid = []
        for j in train_index:
            train_eegid.extend([eegid_all[j]])
        for j in test_index:
            exam_eegid.extend([eegid_all[j]])
        from classification_TE import test2

        probability = test2(train_eegid, exam_eegid)
        probability_tannel.extend(probability)
        print(probability_tannel)
    return probability_tannel


'''
调用classification_TSE.py的test2()进行分类
'''


def TSE_cross_check():
    eegid_all = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U']

    X = np.arange(42).reshape(21, 2)
    probability_tannel = []

    kf = KFold(n_splits=10, shuffle=True)  # 初始化KFold
    for train_index, test_index in kf.split(X):  # 调用split方法切分数据
        print('train_index:%s , test_index: %s ' % (train_index, test_index))
        train_eegid = []
        exam_eegid = []
        for j in train_index:
            train_eegid.extend([eegid_all[j]])
        for j in test_index:
            exam_eegid.extend([eegid_all[j]])
        from classification_TSE import test2

        probability = test2(train_eegid, exam_eegid)
        probability_tannel.extend(probability)
        print(probability_tannel)
    return probability_tannel


if __name__ == '__main__':
    '''
    交叉检验
    '''
    # TE_cross_check()
    # TSE_cross_check()
