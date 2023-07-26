# -*- coding: utf-8 -*-
#
# Title: RTSE的计算
# Author: 寒木春
# Description: 使用随机森林进行计算RTSE
# Refer:
# Date: 2023-07-14
#

import numpy as np
import classification_TSE

'''
计算alp 1.0-2.0的RTSE的值
调用classification_TSE.py的函数
'''


def test1():
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U', 'MA00100V', 'MA00100W', 'MA00100X',
                 'MA00100Y', 'MA00100Z', 'MA001010', 'MA001011', 'MA001012', 'MA001013']

    for i in range(10, 21):
        alp = i / 10.0
        index = ['A1+A2 OFF', '1.10', '1.20', '1.30']
        path = r'./dataset/RTSE/dataset_50/'
        eegid_test = ['MA00100A', 'MA00100B']
        eegid_test1 = eegid_test[0] + str(alp)
        eegid_test2 = eegid_test[1] + str(alp)
        TP1 = np.zeros(16)
        FP1 = np.zeros(16)
        FN1 = np.zeros(16)
        TN1 = np.zeros(16)
        TP_flag, FP_flag, FN_flag, TN_flag = classification_TSE.classificate(path, eegid_test1, eegid_test2, 10, index)
        for i in range(0, len(TP1)):
            TP1[i] = TP1[i] + TP_flag[i]
            FP1[i] = FP1[i] + FP_flag[i]
            TN1[i] = TN1[i] + TN_flag[i]
            FN1[i] = FN1[i] + FN_flag[i]
        """
        输出结果
        """
        # textpath = './dataset/RTSE/result/test'
        # fp1 = open(textpath, 'a', encoding='utf-8')
        # print(f"{alp}", file=fp1)
        for i in range(0, len(TP1)):
            precision = TP1[i] / (TP1[i] + FP1[i])
            recall = TP1[i] / (TP1[i] + FN1[i])
            F1 = 2 * (precision * recall) / (precision + recall)
            print(f"{i * 50 / 32}Hz")
            print(f"precision:{precision}")
            print(f"recall:{recall}")
            print(f"F1:{F1}")
            # print(f"{i * 50 / 32}Hz", file=fp1)
            # print(f"precision:{precision}", file=fp1)
            # print(f"recall:{recall}", file=fp1)
            # print(f"F1:{F1}", file=fp1)


if __name__ == '__main__':
    '''
    计算RTSE的值
    '''
    test1()
