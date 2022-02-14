#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021.9.15

@author: liudongxu
"""

import torch
import os
import librosa
import random
import numpy as np
import linecache
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import json





class CasualUNet_Dataset(Dataset):
    def __init__(self,data_type) -> None:
        super().__init__()
        # self.data_base_dir="/data/liudx/ConferenceSpeechsimulation/data/"
        #下面这个是我们使用的新生成的较大数据集
        # self.data_base_dir="/data/liudx/fardata_large/"
        #我们换成多个噪声源的数据集
        self.data_base_dir="/data/liudx/fardata_large_multinoise/"
        if(data_type=='train'):
            self.data_type=data_type
            self.data_num=28000
            self.data_dir=self.data_base_dir+"train/"
            #for example,a set of wavefiles were saved in /data/liudx/multichannel_libri/MC_Libri_fixed/test/6mic/sample3000/
            #mixture_mic6.wav,spk1_mic6.wav,spk2_mic6.wav
        elif(data_type=='valid'):
            self.data_type=data_type
            self.data_num=8000
            self.data_dir=self.data_base_dir+"valid/"
        elif(data_type=='test'):
            self.data_type=data_type
            self.data_num=4000
            self.data_dir=self.data_base_dir+"test/"

    def __len__(self):
        return self.data_num

    def __getitem__(self,index):
        self.wavedir=self.data_dir+"sample"+str(index+1)+"/"
        # mix_list=[]
        # speech_list=[]
        # noise_list=[]
        # if(self.data_type=='test'):
        #     print("the index is:",index)
        # for i in range(1,7):
        #     mix_wave,sr=librosa.load(self.wavedir+"mixture_mic"+str(i)+".wav",sr=None)
        #     speech_wave,sr=librosa.load(self.wavedir+"speech_mic"+str(i)+".wav",sr=None)
        #     noise_wave,sr=librosa.load(self.wavedir+"noise_mic"+str(i)+".wav",sr=None)
        #     mix_list.append(mix_wave)
        #     speech_list.append(speech_wave)
        #     noise_list.append(noise_wave)
        # mix_batch=np.vstack(mix_list)
        # speech_batch=np.vstack(speech_list)
        # noise_batch=np.vstack(noise_list)
        mix_batch,sr=sf.read(self.wavedir+"mixture.wav")
        speech_batch,sr=sf.read(self.wavedir+"speech.wav")
        # speech_batch,sr=sf.read(self.wavedir+"speech_early.wav",16000)
        #如果需要early reverb而非无混响的纯净语音作为参考，我们可以把对应的speechbatch改一下
        noise_batch,sr=sf.read(self.wavedir+"noise.wav")
        #读取一个方向信息的json
        with open((self.wavedir+"direction_info.json"),'r') as f:
            direction_info=json.load(f)

        #上面读取的维度是(samples,channels),下面需要重整一下
        mix_batch=mix_batch.transpose((1,0))
        speech_batch=speech_batch.transpose((1,0))
        noise_batch=noise_batch.transpose((1,0))
        #重整以后就变成了(channels,samples),和原来一样
        
        speech_ref=speech_batch[0]
        noise_ref=noise_batch[0]
        ref_batch=np.vstack((speech_ref,noise_ref))
        speech_ref=speech_ref[np.newaxis,:]
        noise_ref=noise_ref[np.newaxis,:]

        #加载方向信息
        clean_angle=direction_info['clean_angle']
        # print("clean angle:",clean_angle)
        clean_angle=float(clean_angle)
        # print("clean angle new:",clean_angle)
        # print("clean angle type:",clean_angle.type)
        clean_direction=np.array(clean_angle*np.pi/180.0)
        #一开始是0维的
        clean_direction=clean_direction[np.newaxis]
        clean_direction=clean_direction[np.newaxis,:]
        clean_direction_batch=torch.from_numpy(clean_direction).type(torch.FloatTensor)


        return torch.from_numpy(mix_batch).type(torch.FloatTensor
        ),torch.from_numpy(speech_ref).type(torch.FloatTensor
        ),torch.from_numpy(noise_ref).type(torch.FloatTensor
        ),torch.from_numpy(ref_batch).type(torch.FloatTensor
        )
        # ,torch.from_numpy(spk1_batch).type(torch.FloatTensor
        # ),torch.from_numpy(spk2_batch).type(torch.FloatTensor
        # ),
            # mix_wave=mix_wave[np.newaxis,:,:]
            # spk1_wave=spk1_wave[np.newaxis,:,:]
            # spk2_wave=spk2_wave[np.newaxis,:,:]

            
