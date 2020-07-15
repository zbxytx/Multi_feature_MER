#!/usr/bin/env python
# coding: utf-8

from Local_weighted_mean_register import LWMRegister #face crop and register
from Eulerian_video_magnification import EVM #EVM to magnify motion
from Temporal_interpolation_model import TIM #TIM to interpolate
import Features_extraction as fe #extract LBP-TOP/3DHOG/HOOF
import Classification_and_evaluation as ce #SVM to classify and evaluate

import pandas as pd
import numpy as np
import os
from PIL import Image

def load_data(data_path, df_path, height=480, width=640, data_range=(0, 255)):
    df = pd.read_excel(df_path, keep_default_na=False)
    Subject = df.Subject
    Filename = df.Filename
    OnsetF = df.OnsetFrame
    OffsetF = df.OffsetFrame
    
    data_min, data_max = data_range
    if data_range[0] < 0:
        data_min = 0
    if data_range[1] > len(df):
        data_max = len(df)
    
    data = []
    for i in range(data_min, data_max):
    #for i in range(12):
        ep_data = []
        
        if Subject[i] >= 10:
            ep_path = data_path+'/sub'+str(Subject[i])+'/'+Filename[i]
        else:
            ep_path = data_path+'/sub0'+str(Subject[i])+'/'+Filename[i]

        try:
            images = os.listdir(ep_path)
            images.sort(key=lambda img : int(img[3:-4]))
        except:
            print(ep_path)

        for image_name in images:
            image_path = ep_path+'/'+image_name
            try:
                image_data = np.array(Image.open(image_path).resize((width, height)))
                ep_data.append(image_data)
            except:
                print(image_name)
        ep_data = np.array(ep_data)
        data.append(ep_data)
    data = np.array(data)
    return data

def get_labels(df_path):
    df = pd.read_excel(df_path, keep_default_na=False)
    df = df.drop(['Unnamed: 2', 'Unnamed: 6'], axis=1)

    for i in range(len(df['Estimated Emotion'])):
        if df['Estimated Emotion'][i] == 'repression':
            df['Estimated Emotion'][i] = 'others'

        if df['Estimated Emotion'][i] == 'sadness' or df['Estimated Emotion'][i] == 'fear':
            df['Estimated Emotion'][i] = 'negative'

        if df['Estimated Emotion'][i] == 'disgust':
            df['Estimated Emotion'][i] = 'negative'

        if df['Estimated Emotion'][i] == 'happiness':
            df['Estimated Emotion'][i] = 'positive'
        
    labels = np.array(pd.get_dummies(df['Estimated Emotion']))
    labels = np.array([np.argmax(t) for t in np.array(pd.get_dummies(df['Estimated Emotion']))])
    return labels

def get_uniform_dict(uniform_path):
    uniform=pd.read_csv(uniform_path,sep=' ')
    uniform.columns = ['default', 'uniform']
    uniform.head()
    uniform_dict = {}
    for i in range(len(uniform.default)):
        uniform_dict[uniform.default[i]] = uniform.uniform[i]
    return uniform_dict

if __name__ == '__main__':
    data_path = 'CASME II'
    df_path = 'CASME II/CASME2.xlsx'
    predictor_path = 'CASME II/shape_predictor_68_face_landmarks.dat'
    uniform_path = 'CASME II/UniformLBP8.txt'
    
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists('result/features'):
        os.mkdir('result/features')
    lwm_result_path = 'result/lwm_result'
    feature_path = 'result/features'
    
    df = pd.read_excel(df_path, keep_default_na=False)
    
    print("load data and crop and register")
    #crop and register
    #data[0][0]: standard face
    lwm = LWMRegister(load_data(data_path, df_path, data_range=(0, 1))[0][0], predictor_path,width=192, height=192) 
    lwm_result = []
    for i in range(len(df)//10 + 1):
        data = load_data(data_path, df_path, data_range=(i*10, (i+1)*10))
        for seq in data:
            lwm_result.append(lwm.run(seq, aligned=False))
    lwm_result = np.array(lwm_result)
    np.save(lwm_result_path, lwm_result)
    del data
        
    print("motion magnification")
    #motion magnification
    evm = EVM(fps=200, low=0.2, high=2.4, level=6, alpha=8, lam_c=16, iq_reduce=0.1)
    evm_result = []
    for seq in lwm_result:
        evm_result.append(evm.run(seq))
    del lwm_result
        
    print("temporal interpolate")
    #temporal interpolate
    tim = TIM()
    tim_result = []
    for seq in evm_result:
        tim_result.append(tim.run(seq, 10)) #target frames
    del evm_result
    
    print("feature extraction")
    #features extraction
    LBP_feature = []
    HOG_feature = []
    HOOF_feature = []
    for seq in tim_result:
        LBP_feature.append(fe.get_ep_features(seq, uniform_dict = get_uniform_dict(uniform_path), feature='LBP-TOP', 
                                              t_times=1, x_times=5, y_times=5))
        HOG_feature.append(fe.get_ep_features(seq, feature='3DHOG', t_times=5, x_times=5, y_times=5))
        HOOF_feature.append(fe.get_ep_features(seq, feature='HOOF', t_times=5, x_times=5, y_times=5))
    LBP_feature = np.array(LBP_feature)
    HOG_feature = np.array(HOG_feature)
    HOOF_feature = np.array(HOOF_feature)
    np.save(feature_path+'/LBP_feature', LBP_feature)
    np.save(feature_path+'/HOG_feature', HOG_feature)
    np.save(feature_path+'/HOOF_feature', HOOF_feature)
    del tim_result
    
    print("evaluation")
    #evaluation
    labels = get_labels(df_path)
    sub_list = df.Subject
    print('LBP_features:')
    ce.get_best_average(LBP_feature, labels, sub_list, kernel='linear', split='loso', average='macro')
    ce.get_best_average(LBP_feature, labels, sub_list, kernel='poly', split='loso', average='macro')
    ce.get_best_average(LBP_feature, labels, sub_list, kernel='rbf', split='loso', average='macro')
    print('HOG_features:')
    ce.get_best_average(HOG_feature, labels, sub_list, kernel='linear', split='loso', average='macro')
    ce.get_best_average(HOG_feature, labels, sub_list, kernel='poly', split='loso', average='macro')
    ce.get_best_average(HOG_feature, labels, sub_list, kernel='rbf', split='loso', average='macro')
    print('HOOF_features:')
    ce.get_best_average(HOOF_feature, labels, sub_list, kernel='linear', split='loso', average='macro')
    ce.get_best_average(HOOF_feature, labels, sub_list, kernel='poly', split='loso', average='macro')
    ce.get_best_average(HOOF_feature, labels, sub_list, kernel='rbf', split='loso', average='macro')


