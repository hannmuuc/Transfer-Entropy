# -*- coding: utf-8 -*-

#
# Title: 语言任务预处理
# Author: 程勇老师
# Description:
# Refer: Cheng Yong 对语言任务进行预处理的Pipeline
# Date: 2023-07-11
#


import os
import h5py
import mne
from mne.preprocessing import ICA


#
# 保存多个evoked为h5文件.
#
def write_evoked(h5path, evename_list, evoked_list):
    '''
    :param filename: 保存文件名
    :param evename_list: 关键字列表
    :param evoked_list: 关键字对应的evoked列表
    :return:
    '''
    assert len(evename_list) == len(evoked_list)
    with h5py.File(h5path, 'a') as f:
        for i in range(len(evename_list)):
            new_group = f.create_group(evename_list[i])
            new_group.create_dataset('value', data=evoked_list[i])


def create_evoked_h5(eegid):
    edf_dirpath = r'C:\Users\asus\Desktop\robetstudy\langtask\edf'
    h5path = os.path.join(r'./evoked', eegid + '.h5')
    if os.path.isfile(h5path):
        print('hmc')
        return
    # Step1: 读入edf格式数据
    raw = mne.io.read_raw_edf(edf_dirpath + "\\" + eegid + ".edf", preload=True)
    # raw.crop(tmin=3.)      # 去除前三秒校正信号
    chan_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
                 'Cz', 'Pz', 'A1', 'A2', 'M1', 'M2']
    raw.pick_channels(chan_list)  # 挑选通道
    montage = mne.channels.make_standard_montage("standard_1020")  # 使用规范导联名
    raw.set_montage(montage)
    data = raw.get_data()
    num_chan = data.shape[0]
    num_sample = data.shape[1]
    sfreq = raw.info['sfreq']
    # print(num_chan)
    # print(num_sample)
    # print(sfreq)

    # Step2: 进行工频滤波
    raw = raw.notch_filter(freqs=(50))

    # Step3: 进行带通滤波
    raw = raw.filter(l_freq=0.1, h_freq=100)

    # Step4: 去伪影
    ica = ICA(max_iter='auto')
    raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)  # ⾼通1Hz的数据进⾏ICA及相关成分剔除，再应⽤到⾼通0.1Hz的数据上
    ica.fit(raw_for_ica)
    muscle_idx_auto, scores = ica.find_bads_muscle(raw_for_ica)  # 自动选择肌肉伪影
    ica.exclude = muscle_idx_auto
    ica.apply(raw)

    # Step5: 重参考
    raw.set_eeg_reference(ref_channels=['A1', 'A2'])

    # Step6: 获得Epoch信号
    events = raw.annotations
    event_descriptions = events.description  # 获取事件描述
    onset_times = events.onset  # 获取事件开始时间
    # 遍历每个事件，获取对应的信号数据
    evename_list = ['A1+A2 OFF', '1.10', '1.20', '1.30', '2.10', '2.20', '2.30', '3.10', '4.10', '5.10', '6.10', '6.20',
                    '6.30']
    evedata_list = []
    for i in range(len(event_descriptions) - 1):
        event_start = onset_times[i]
        next_event_start = onset_times[i + 1]
        event_signal = raw.copy().crop(tmin=event_start, tmax=next_event_start)
        if event_descriptions[i].strip() in evename_list:
            evedata_list.append(event_signal.get_data())
        # 处理两个事件之间的信号数据（event_signal）
        # 处理事件对应的信号数据（event_signal）
        # print(event_signal[0][0].shape)
        # print("----")

    print(len(evedata_list))
    # Step7: 保存数据为h5格式
    write_evoked(h5path, evename_list, evedata_list)


# 主函数
if __name__ == '__main__':
    all_eegid = ['MA00100A', 'MA00100B', 'MA00100C', 'MA00100D', 'MA00100E', 'MA00100F', 'MA00100G', 'MA00100H',
                 'MA00100I', 'MA00100J', 'MA00100K', 'MA00100L', 'MA00100M', 'MA00100N', 'MA00100O', 'MA00100P',
                 'MA00100Q', 'MA00100R', 'MA00100S', 'MA00100T', 'MA00100U', 'MA00100V', 'MA00100W', 'MA00100X',
                 'MA00100Y', 'MA00100Z', 'MA001010', 'MA001011', 'MA001012', 'MA001013']
    for eegid in all_eegid:
        create_evoked_h5(eegid)
