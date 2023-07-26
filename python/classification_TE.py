# -*- coding: utf-8 -*-
#
# Title: 使用随机森林进行分析
# Author: 寒木春
# Description: 使用随机森林进行分析
# Refer:
# Date: 2023-07-14
#

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# 定义随机森林分类方法
def random_forest_classification(matrix_data, labels):
    # 将矩阵数据转换为特征向量形式
    # matrix_data为一个包含多个矩阵的列表，每个矩阵可以是numpy数组或列表等形式
    features = []
    for matrix in matrix_data:
        # 对于每个矩阵，可以根据需要提取特定的特征向量表示
        # 这里假设直接使用矩阵的扁平化形式作为特征向量
        feature_vector = matrix.flatten()
        features.append(feature_vector)
    features = np.array(features)

    # 构建并训练随机森林分类器
    classifier = RandomForestClassifier(n_estimators=50, max_features=50)
    classifier.fit(features, labels)

    return classifier


def read_multiple_matrices_file(file_path):
    matrices = []
    matrices_name = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        num_matrices = len(lines) // 24  # 每个矩阵包含24行（1行名称 + 23行数据）
        for i in range(num_matrices):
            matrix_lines = lines[i * 24: (i + 1) * 24]  # 提取当前矩阵的所有行

            # 获取当前矩阵的名称
            name = matrix_lines[0].strip()
            # 逐行读取矩阵数据并转换为二维数组
            matrix_data = []
            for line in matrix_lines[1:]:
                row1 = line.strip().split(',')
                row = row1[0:23]
                row_data = [float(value) for value in row]
                matrix_data.append(row_data)

            matrix_data = np.array(matrix_data)
            '''
            归一化
            '''
            # _range = np.max(matrix_data) - np.min(matrix_data)
            # matrix_data=(matrix_data - np.min(matrix_data)) / _range

            '''
            返回
            '''
            matrices.append(matrix_data)
            matrices_name.append(name)

    return matrices_name, matrices


def train(path, eegid, matrices_all, labels_all, index):
    eegid_txt = eegid
    textpath = os.path.join(path, eegid_txt)
    matrices_name, matrices = read_multiple_matrices_file(textpath)

    for i in range(0, len(matrices)):
        if matrices_name[i] in index:
            matrices_all = matrices_all + [matrices[i]]
            labels_all = labels_all + [matrices_name[i][0]]
    return matrices_all, labels_all


def exam(path, eegid, classifier, index):
    eegid_txt = eegid
    textpath = os.path.join(path, eegid_txt)
    matrices_name_test, matrices_test = read_multiple_matrices_file(textpath);
    print(matrices_name_test)
    flag_right = 0
    flag_wrong = 0
    for i in range(0, len(matrices_test)):
        if matrices_name_test[i] in index:
            new_feature_vector = matrices_test[i].flatten()
            predicted_label = classifier.predict([new_feature_vector])
            if predicted_label[0][0] == matrices_name_test[i][0]:
                flag_right = flag_right + 1
            else:
                flag_wrong = flag_wrong + 1
            # print(f'predict:{predicted_label[0]} actual:{matrices_name_test[i]}')

    print(f'Right:{flag_right} Wrong:{flag_wrong}')
    return flag_right, flag_wrong


def exam_confusion(path, eegid, classifier, index):
    eegid_txt = eegid
    textpath = os.path.join(path, eegid_txt)
    matrices_name_test, matrices_test = read_multiple_matrices_file(textpath)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    probability_all = []
    # 计算每个通道有几个数据
    unit_test = len(matrices_test) / 13

    # 计算概率
    probability_aLL_unit = []
    probability_flag = 0
    probability_unit = 0
    probability_unit_sum = 0
    probability_truth = 0

    for i in range(0, len(matrices_test)):
        if matrices_name_test[i] in index:
            new_feature_vector = matrices_test[i].flatten()
            predicted_label = classifier.predict([new_feature_vector])
            predicted_tool = classifier.predict_proba([new_feature_vector])
            true_flag = 0

            # 有通道
            probability_flag = 1
            probability_unit_sum = probability_unit_sum + 1
            probability_unit = probability_unit + predicted_tool[0][0]

            if predicted_label[0][0] == matrices_name_test[i][0]:
                if predicted_label[0][0] == 'A':
                    TN = TN + 1
                else:
                    TP = TP + 1
                    true_flag = 1
                    probability_truth = 1
            else:
                if predicted_label[0][0] == 'A':
                    FN = FN + 1
                    true_flag = 1
                    probability_truth = 1
                else:
                    FP = FP + 1

            result = [predicted_tool[0][0], true_flag]
            probability_all.extend([result])
        # 每判断完一个通道
        if i % unit_test == unit_test - 1 and probability_flag == 1:
            result23 = [probability_unit / probability_unit_sum, probability_truth]
            probability_aLL_unit.extend([result23])
            probability_flag = 0
            probability_unit = 0
            probability_unit_sum = 0
            probability_truth = 0

    return TP, FP, FN, TN, probability_all, probability_aLL_unit


def exam_test3(path, eegid, classifier, index):
    TP = np.zeros(60)
    FP = np.zeros(60)
    FN = np.zeros(60)
    TN = np.zeros(60)
    eegid_txt = eegid
    textpath = os.path.join(path, eegid_txt)
    matrices_name_test, matrices_test = read_multiple_matrices_file(textpath)
    for i in range(0, len(matrices_test)):
        if matrices_name_test[i] in index:
            new_feature_vector = matrices_test[i].flatten()
            predicted_label = classifier.predict([new_feature_vector])
            if predicted_label[0][0] == matrices_name_test[i][0]:
                if predicted_label[0][0] == 'A':
                    TP[i % 60] = TN[i % 60] + 1
                else:
                    TP[i % 60] = TP[i % 60] + 1
            else:
                if predicted_label[0][0] == 'A':
                    FN[i % 60] = FN[i % 60] + 1
                else:
                    FP[i % 60] = FP[i % 60] + 1

    return TP, FP, FN, TN


'''
初步实验F1值 不为最终结果
'''


def test1():
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U', 'MA00100V', 'MA00100W', 'MA00100X',
                 'MA00100Y', 'MA00100Z', 'MA001010', 'MA001011', 'MA001012', 'MA001013']

    '''
    训练集和测试集
    '''
    train_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                   'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                   'MA00100R']
    exam_eegid = ['MA00100Q', 'MA00100S']
    matrices = []
    labels = []
    '''
    数据路径
    '''
    path = r'./dataset/TE/dataset_50/'
    '''
    训练随机森林
    '''
    # 0-5000
    index = ['A1+A2 OFF', '1.10', '1.20', '1.30']
    for eegid in train_eegid:
        print(eegid)
        matrices, labels = train(path, eegid, matrices, labels, index)
    # 5000-8000
    path1 = r'./dataset/TE/dataset_50_test/'
    for eegid in train_eegid:
        print(eegid)
        matrices, labels = train(path1, eegid, matrices, labels, index)
    classifier = random_forest_classification(matrices, labels)
    '''
    测试随机森林
    '''
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    probability_all = []
    probability_tannel = []
    for eegid in exam_eegid:
        TP_flag, FP_flag, FN_flag, TN_flag, probability, probability_tannel = exam_confusion(path, eegid, classifier,
                                                                                             index)
        TP = TP + TP_flag
        FP = FP + FP_flag
        FN = FN + FN_flag
        TN = TN + TN_flag
        probability_all = probability_all + probability

    print(TP, FP, FN, TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    print(f"precision:{precision}")
    print(f"recall:{recall}")
    print(f"F1:{F1}")
    return probability_all


'''
test2()
对外实现的接口
被cross_check()调用
'''


def test2(train_eegid, exam_eegid):
    matrices = []
    labels = []
    '''
    数据路径
    '''
    path = r'./dataset/TE/dataset_50/'
    '''
    训练随机森林
    '''
    # 0-5000
    index = ['A1+A2 OFF', '6.10']
    for eegid in train_eegid:
        print(eegid)
        matrices, labels = train(path, eegid, matrices, labels, index)
    # 5000-8000
    path1 = r'./dataset/TE/dataset_50_test/'
    for eegid in train_eegid:
        print(eegid)
        matrices, labels = train(path1, eegid, matrices, labels, index)
    classifier = random_forest_classification(matrices, labels)
    '''
    测试随机森林
    '''
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    probability_all = []
    probability_tannel = []
    for eegid in exam_eegid:
        TP_flag, FP_flag, FN_flag, TN_flag, probability, probability_tannel = exam_confusion(path, eegid, classifier,
                                                                                             index)
        TP = TP + TP_flag
        FP = FP + FP_flag
        FN = FN + FN_flag
        TN = TN + TN_flag
        probability_all = probability_all + probability

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    print(f"precision:{precision}")
    print(f"recall:{recall}")
    print(f"F1:{F1}")
    return probability_tannel


'''
第三种方法计算
'''


def test3():
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U', 'MA00100V', 'MA00100W', 'MA00100X',
                 'MA00100Y', 'MA00100Z', 'MA001010', 'MA001011', 'MA001012', 'MA001013']
    eegid_test = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                  'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                  'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U']

    TP = np.zeros(60)
    FP = np.zeros(60)
    FN = np.zeros(60)
    TN = np.zeros(60)
    for eegid in eegid_test:
        # 数据路径
        path1 = r'./dataset/TE/dataset_50/'
        path2 = r'./dataset/TE/dataset_50_test/'
        # 训练随机森林
        index = ['A1+A2 OFF', '1.20']
        matrices = []
        labels = []
        matrices, labels = train(path1, eegid, matrices, labels, index)
        classifier = random_forest_classification(matrices, labels)
        # 测试随机森林

        TP_flag, FP_flag, FN_flag, TN_flag = exam_test3(path2, eegid, classifier, index)
        for i in range(0, len(TP)):
            TP[i] = TP[i] + TP_flag[i]
            FP[i] = FP[i] + FP_flag[i]
            TN[i] = TN[i] + TN_flag[i]
            FN[i] = FN[i] + FN_flag[i]

    for i in range(0, len(TP)):
        precision = TP[i] / (TP[i] + FP[i])
        recall = TP[i] / (TP[i] + FN[i])
        F1 = 2 * (precision * recall) / (precision + recall)
        print(f"precision:{precision}")
        print(f"recall:{recall}")
        print(f"F1:{F1}")
        print('')


if __name__ == '__main__':
    '''
    初步实验F1值 不为最终结果
    '''
    test1()
    '''
    第三种实验
    '''
    # test3()
