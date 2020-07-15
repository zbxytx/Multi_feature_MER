#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import cv2
import scipy.signal as signal
import scipy.fftpack as fftpack
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class EVM:
    def __init__(self, fps=200, low=0.2, high=2.4, level=6, alpha=8, lam_c=16, iq_reduce=0.1):
        self.fps = fps
        self.low = low
        self.high = high
        self.level = level
        self.alpha = alpha
        self.lam_c = lam_c
        self.iq_reduce = iq_reduce
        
    #Build Gaussian Pyramid
    def build_gaussian_pyramid(self, src,level=3):
        s=src.copy()
        pyramid=[s]
        for i in range(level):
            s=cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid

    #Build Laplacian Pyramid
    def build_laplacian_pyramid(self, src,level=3):
        gaussianPyramid = self.build_gaussian_pyramid(src, level)
        pyramid=[]
        for i in range(level,0,-1):
            GE=cv2.pyrUp(gaussianPyramid[i])
            L=cv2.subtract(gaussianPyramid[i-1],GE)
            pyramid.append(L)
        return pyramid

    #build laplacian pyramid for video
    def laplacian_video(self, video_tensor,level=3):
        tensor_list=[]
        for i in range(0,video_tensor.shape[0]):
            frame=video_tensor[i]
            pyr=self.build_laplacian_pyramid(frame,level=level)
            if i==0:
                for k in range(level):
                    tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
            for n in range(level):
                tensor_list[n][i] = pyr[n]
        return tensor_list

    #butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        omega = 0.5 * fs
        low = lowcut / omega
        high = highcut / omega
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.lfilter(b, a, data, axis=0)
        return y

    #reconstract video from laplacian pyramid
    def reconstract_from_tensorlist(self, filter_tensor_list,level=3):
        final=np.zeros(filter_tensor_list[-1].shape)
        for i in range(filter_tensor_list[0].shape[0]):
            up = filter_tensor_list[0][i]
            for n in range(level-1):
                up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]#可以改为up=cv2.pyrUp(up)
            final[i]=up
        return final

    #change color space
    def rgb2yiq(self, image):
        return np.array(tf.image.rgb_to_yiq(image.astype('float32')))

    def yiq2rgb(self, image):
        return np.array(tf.image.yiq_to_rgb(image.astype('float32')))
    
    #manify motion
    def magnify_motion(self, img_seq, fps, low, high, level=6, alpha=8, lam_c=16, iq_reduce = 0.1):
    
        #将图像序列转为yiq空间
        t = []
        height = img_seq[0].shape[0]
        width = img_seq[0].shape[1]
        temp = 2**level
        for i in range(len(img_seq)):
            #防止width与height不能整除temp，先resize
            t.append(self.rgb2yiq(cv2.resize(img_seq[i],((width//temp)*temp, (height//temp)*temp),interpolation=cv2.INTER_CUBIC)))
            #t.append(self.rgb2yiq(img_seq[i]).astype('float32'))
        t = np.array(t)
        f = fps
    
        #注：此处使用seq与seq[0]的差值进行滤波
        lap_video_list=self.laplacian_video(t-t[0],level=level)
        filter_tensor_list=[]
        for i in range(level):
            filter_tensor=self.butter_bandpass_filter(lap_video_list[i],low,high,f)
        
            height = filter_tensor.shape[1]
            width = filter_tensor.shape[2]
            delta = lam_c / 8.0 / (1.0 + alpha)
            lam = math.sqrt(width * width + height * height) / 3
            cur_alpha = lam / delta / 8 - 1
        
            if i ==0 or i == level-1:
                filter_tensor *= 0
            else:
                filter_tensor *= min(alpha, cur_alpha)
            filter_tensor_list.append(filter_tensor)
        
        recon=self.reconstract_from_tensorlist(filter_tensor_list, level=level)
        recon[..., 1] *= iq_reduce
        recon[..., 2] *= iq_reduce
    
        final=t+recon
        final = np.array(final)
        for i in range(len(final)):
            #final[i] = yiq2rgb(final[i])
            final[i] = self.yiq2rgb(cv2.resize(final[i],(width, height),interpolation=cv2.INTER_CUBIC))
    
        #防止数值超出[0, 255]
        final[final<18] = (18-final[final<18])/(18-np.min(final))*18
        final[final > 238] = 238+(final[final > 238]-238)/(np.max(final)-238)*17
    
        return final
    
    def run(self, img_seq):
        return self.magnify_motion(img_seq, self.fps, self.low, self.high, self.level, self.alpha, self.lam_c, self.iq_reduce).astype('int')

