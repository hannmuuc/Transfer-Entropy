# -*- coding: utf-8 -*-
#
# Title: TE.dll接口函数
# Author: 计科2001-韩明辰
# Description: Entropy类实现TE.dll的接口
# Refer:
# Date: 2023-07-10
#

import ctypes
from ctypes import *
import numpy as np


class StructPointer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_double * 550),
                ("num", ctypes.c_int)]


class Entropy:
    def __init__(self, data_first, data_second):
        self.data_first = data_first
        self.data_second = data_second

    '''
    调用TE.dll
    '''

    @staticmethod
    def calculate_average(data, flag):
        result = []
        for i in range(0, len(data), flag):
            subset = data[i:i + flag]
            average = sum(subset) / len(subset)
            result.append(average)
        return result

    def prework(self):
        dll = CDLL(r'./dll/TE.dll')
        date_len = len(self.data_first)
        if date_len != len(self.data_second):
            return -1
        first_parameter = (ctypes.c_double * date_len)()
        second_parameter = (ctypes.c_double * date_len)()
        for i in range(0, date_len):
            first_parameter[i] = self.data_first[i]
            second_parameter[i] = self.data_second[i]
        num = ctypes.c_int(date_len)
        return first_parameter, second_parameter, num, dll

    '''
    计算转移熵
    '''

    def TE(self, delay, k, l):

        (first_parameter, second_parameter, num, dll) = self.prework()
        '''
        TE函数类型
        c++接口 double TE(double* X, double* Y, int num, int delay, int k, int l)
        '''
        date_te = dll.TE
        date_te.argtypes = (
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int)
        date_te.restype = ctypes.c_double

        te_res = date_te(first_parameter, second_parameter, num, delay, k, l)

        return te_res

    '''
    计算广义熵
    '''

    def RTE(self, delay, k, l, alp):
        (first_parameter, second_parameter, num, dll) = self.prework()
        '''
        TE函数类型
        c++接口 double RTE(double* X, double* Y, int num, int delay, int k, int l,double alp)
        数组 数组 int int int int
        '''
        date_rte = dll.RTE
        date_rte.argtypes = (
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_double)
        date_rte.restype = ctypes.c_double

        te_res = date_rte(first_parameter, second_parameter, num, delay, k, l, alp)

        return te_res

    '''
    转移谱熵
    '''

    def TSE(self, delay, k, l):
        (first_parameter, second_parameter, num, dll) = self.prework()

        """
        TSE函数
        c++接口 Node_pointer TSE(double* X, double* Y, int num, int delay, int k, int l);
        """
        date_tse = dll.TSE

        date_tse.argtypes = (
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int)
        date_tse.restype = ctypes.POINTER(StructPointer)

        tse_res = date_tse(first_parameter, second_parameter, num, delay, k, l)
        list1 = []
        tse_res_num = tse_res.contents.num
        for i in range(0, tse_res_num):
            list1.append(tse_res.contents.data[i])

        return list1

    '''
    广义谱熵 
    '''

    def RTSE(self, delay, k, l, alp):
        (first_parameter, second_parameter, num, dll) = self.prework()
        '''
        RTSE函数
        c++接口 StructPointer RTSE(double* X, double* Y, int num, int delay, int k, int l, double alp);
        '''
        date_rtse = dll.RTSE
        date_rtse.argtypes = (
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_double)
        date_rtse.restype = ctypes.POINTER(StructPointer)

        rtse_res = date_rtse(first_parameter, second_parameter, num, delay, k, l, alp)

        list1 = []
        rtse_res_num = rtse_res.contents.num
        for i in range(0, rtse_res_num):
            list1.append(rtse_res.contents.data[i])

        return list1


class Entropy_thread:
    def __init__(self, data):
        self.data = data

    def print_data(self):
        fp1 = open(r'./dataset/data.txt', 'w', encoding='utf-8')
        date_len = len(self.data)
        length_item = len(self.data[0])
        print(date_len, file=fp1, end=' ')
        print(length_item, file=fp1)
        np.set_printoptions(suppress=True)
        for i in range(0, date_len):
            for j in range(0, length_item):
                print("{:.8f}".format(self.data[i][j]), file=fp1, end=' ')

    def prework(self):
        dll = CDLL(r'./dll/TE.dll')
        date_len = len(self.data)
        length_item = len(self.data[0])
        first_parameter = (ctypes.c_double * (length_item * date_len))()
        for i in range(0, date_len):
            for j in range(0, length_item):
                first_parameter[i * length_item + j] = self.data[i][j]

        return first_parameter, date_len, length_item, dll

    def TE_matric(self, delay, k, l, alp):
        (first_parameter, cow, row, dll) = self.prework()
        print(cow, row)
        '''
        Node_pointer TE_matric(double* X, int cow, int row, int delay, int k, int l, int alp);
        '''
        date_tem = dll.TE_matric
        date_tem.argtypes = (
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_double)
        date_tem.restype = ctypes.POINTER(StructPointer)

        print("awd")
        rtse_res = date_tem(first_parameter, cow, row, delay, k, l, alp)

        '''
        变成矩阵
        '''
        list1 = []
        rtse_res_num = rtse_res.contents.num
        for i in range(0, rtse_res_num):
            list1.append(rtse_res.contents.data[i])
        b = np.array(list1).reshape(cow, cow)
        return b
