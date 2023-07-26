# -*- coding: utf-8 -*-
#
# Title: 使用决策树进行分析
# Author: 寒木春
# Description: 使用决策树进行频谱分析
# Refer:
# Date: 2023-07-16

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def reshape_matrix(matrix):
    num_rows, num_cols = matrix.shape
    reshaped_matrix = np.zeros((num_cols, 23, 23))

    for col in range(num_cols):
        flattened_col = matrix[:, col]
        reshaped_col = flattened_col.reshape((23, 23))
        reshaped_matrix[col] = reshaped_col

    return reshaped_matrix


def TSE_read_multiple_matrices_file(file_path, interval):
    matrices = [[] for _ in range(interval)]
    matrices_name = [[] for _ in range(interval)]
    with open(file_path, 'r') as file:
        lines = file.readlines()

        num_matrices = len(lines) // 530  # 每个矩阵包含24行（1行名称 + 23行数据）

        for i in range(num_matrices):
            matrix_lines = lines[i * 530: (i + 1) * 530]  # 提取当前矩阵的所有行

            # 获取当前矩阵的名称
            name = matrix_lines[0].strip()

            # 逐行读取矩阵数据并转换为二维数组
            matrix_data = []
            for line in matrix_lines[1:]:
                row1 = line.strip().split(',')
                row = row1[0:16]
                row_data = [float(value) for value in row]
                matrix_data.append(row_data)

            matrix_data = np.array(matrix_data)
            reshaped_matrix = reshape_matrix(matrix_data)
            # 每个元素为一个三维矩阵
            matrices[i % interval].append(reshaped_matrix)
            matrices_name[i % interval].append(name)

    return matrices, matrices_name


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
    classifier = RandomForestClassifier()
    classifier.fit(features, labels)

    return classifier


def train(matrices_train, matrices_name_train, start, end, name_index, frequent):
    matrices_train_res = []
    labels_name = []
    for i in range(start, end):
        for j in range(0, len(matrices_train[i])):
            if matrices_name_train[i][j] in name_index:
                matrices_train_res.append(matrices_train[i][j][frequent])
                labels_name.append(matrices_name_train[i][j])
    return matrices_train_res, labels_name


def exam(matrices_test, matrices_name_test, classifier, start, end, index, frequent):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    print(len(matrices_test))
    for i in range(start, end):
        for j in range(0, len(matrices_test[i])):
            if matrices_name_test[i][j] in index:
                new_feature_vector = matrices_test[i][j][frequent].flatten()
                predicted_label = classifier.predict([new_feature_vector])
                # 判断对错
                if predicted_label[0][0] == matrices_name_test[i][j][0]:
                    if predicted_label[0][0] == 'A':
                        TN = TN + 1
                    else:
                        TP = TP + 1
                else:
                    if predicted_label[0][0] == 'A':
                        FN = FN + 1
                    else:
                        FP = FP + 1

    return TP, FP, FN, TN


'''
进行分类
'''


def classificate(path, eegid, eegid_test, data_num, index):
    file_path = os.path.join(path, eegid)
    file_test = os.path.join(path, eegid_test)
    # matrices为(50,13,(16,23,232))的list matrices_name为(50,13,‘string’)
    matrices_all, matrices_name_all = TSE_read_multiple_matrices_file(file_path, data_num)
    matrices_all_test, matrices_name_all_test = TSE_read_multiple_matrices_file(file_test, data_num)

    print(len(matrices_all))
    TP = np.zeros(16)
    FP = np.zeros(16)
    FN = np.zeros(16)
    TN = np.zeros(16)
    matrices_train = []
    labels = []
    for frequent in range(0, 16):
        # 训练
        matrices_train_1, labels_2 = train(matrices_all, matrices_name_all, 0, data_num, index, frequent)
        matrices_train.extend(matrices_train_1)
        labels.extend(labels_2)
        # 训练
        classifier = random_forest_classification(matrices_train, labels)
        # 测试
        TP_flag, FP_flag, FN_flag, TN_flag = exam(matrices_all_test, matrices_name_all_test, classifier, 0, data_num,
                                                  index,
                                                  frequent)
        TP[frequent] = TP[frequent] + TP_flag
        FP[frequent] = FP[frequent] + FP_flag
        FN[frequent] = FN[frequent] + FN_flag
        TN[frequent] = TN[frequent] + TN_flag

    return TP, FP, FN, TN


'''
实现的对外接口
进行频谱的机器学习
'''


def test2(train_eegid, exam_eegid):
    matrices = []
    labels = []
    '''
    数据路径
    '''
    path = r'./dataset/TSE/dataset_50_test/'
    '''
    训练随机森林
    '''
    # 0-5000
    index = ['A1+A2 OFF', '2.10', '2.20', '2.30']

    from classification_TE import train

    for eegid in train_eegid:
        print(eegid)
        matrices, labels = train(path, eegid, matrices, labels, index)

    from classification_TE import random_forest_classification
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

    from classification_TE import exam_confusion

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


def TSE_test():
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U', 'MA00100V', 'MA00100W', 'MA00100X',
                 'MA00100Y', 'MA00100Z', 'MA001010', 'MA001011', 'MA001012', 'MA001013']
    path = r'./dataset/TSE/dataset_50/'
    eegid_test = ['MA00100A', 'MA00100B', 'MA00100C']
    index = ['A1+A2 OFF', '6.10', '6.20', '6.30']
    TP1 = np.zeros(16)
    FP1 = np.zeros(16)
    FN1 = np.zeros(16)
    TN1 = np.zeros(16)
    TP_flag, FP_flag, FN_flag, TN_flag = classificate(path, eegid_test[0], eegid_test[1], 50, index)
    for i in range(0, len(TP1)):
        TP1[i] = TP1[i] + TP_flag[i]
        FP1[i] = FP1[i] + FP_flag[i]
        TN1[i] = TN1[i] + TN_flag[i]
        FN1[i] = FN1[i] + FN_flag[i]
    '''
        for eegid in eegid_test:
        right_flag, wrong_flag = classificate(path, eegid)
        right_list = right_list + right_flag
        wrong_list = wrong_list + wrong_flag
    '''

    for i in range(0, len(TP1)):
        precision = TP1[i] / (TP1[i] + FP1[i])
        recall = TP1[i] / (TP1[i] + FN1[i])
        F1 = 2 * (precision * recall) / (precision + recall)
        print(f"{i * 50 / 32}Hz")
        print(f"precision:{precision}")
        print(f"recall:{recall}")
        print(f"F1:{F1}")


if __name__ == '__main__':
    '''
    初步实现计算F1值
    验证代码
    不为最终结果
    '''
    TSE_test()
