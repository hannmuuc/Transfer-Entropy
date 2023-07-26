# -*- coding: utf-8 -*-
#
# Title: 提取脑电图特征
# Author: 寒木春
# Description: 使用接口提取脑电图特征
# Refer:
# Date: 2023-07-17
#

import characteristics_interface


def get_TE():
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U']
    path = r'dataset/TE/dataset_50_test/'
    for eegid in all_eegid:
        characteristics_interface.TE_create_data_txt(path, eegid)

    # show(all_eegid[0], all_eegid[1], 1000)


def get_TSE():
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U']
    path = r'dataset/TSE/dataset_50_test/'
    for eegid in all_eegid:
        characteristics_interface.TSE_create_data_txt(path, eegid)


def get_RTE():
    all_eegid = ['MA00100A', 'MA00100B']
    path = r'dataset/RTE/dataset_50/'
    for eegid in all_eegid:
        characteristics_interface.RTE_create_data_txt(path, eegid)


def get_RTSE():
    all_eegid = ['MA00100A', 'MA00100B']
    path = r'dataset/RTSE/dataset_50/'
    import gc
    for eegid in all_eegid:
        characteristics_interface.RTSE_create_data_txt(path, eegid)
        gc.collect()


if __name__ == '__main__':
    '''
    生成TE的部分数据
    '''
    # get_TE()
    '''
    生成TSE部分数据
    '''
    # get_TSE()
    '''
    生出RTE的部分数据
    '''
    #get_RTE()
    '''
    生成RTSE的部分数据
    '''
    #get_RTSE()
