#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold,StratifiedKFold

#sub_list: sub_list[i]==j -> 数据i属于subject j
#target: target subject number
def data_split(data, label, sub_list, target):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    for i in range(len(data)):
        if sub_list[i] == target:
            test_data.append(data[i])
            test_label.append(label[i])
        else:
            train_data.append(data[i])
            train_label.append(label[i])
    
    return np.array(train_data), np.array(test_data), np.array(train_label), np.array(test_label)

def avg_score(data, label, sub_list, kernel='rbf', C=2, gamma=10, degree=3, decision_function_shape='ovr',
              n_splits=10, split='loso', seed=7, average='macro'):
    
    classifier = svm.SVC(C=C,degree=degree,kernel=kernel,gamma=gamma,decision_function_shape=decision_function_shape)
    score_list = []
    result_list = []
    real_list = []
    f1_list = []
    if split=='loso':
        for i in range(1, 27):
            train_data,test_data,train_label,test_label = data_split(data, label, sub_list, i)
            classifier.fit(train_data,train_label.ravel())
            result = classifier.predict(test_data)
            for rst in result:
                result_list.append(rst)
            for real in test_label:
                real_list.append(real)
            score_list.append(classifier.score(test_data,test_label))
            f1_list.append(metrics.f1_score(test_label, result, average = average))
    else:
        SKF=StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=seed)
        for train_index, test_index in SKF.split(data,label):
            train_data = data[train_index]
            train_label = label[train_index]
            test_data = data[test_index]
            test_label = label[test_index]
            
            classifier.fit(train_data,train_label.ravel())
            result = classifier.predict(test_data)
            for rst in result:
                result_list.append(rst)
            for real in test_label:
                real_list.append(real)
            #print('real:\t', test_label)
            #print('result:\t', result)
            score_list.append(classifier.score(test_data,test_label))
            f1_list.append(metrics.f1_score(test_label, result, average = average))
    
    acc_score = metrics.accuracy_score(real_list, result_list)
    f1_score = metrics.f1_score(real_list, result_list, average = average)

    return np.mean(score_list), acc_score, f1_score, np.mean(f1_list)

def get_best_average(data, labels, sub_list, kernel='linear', split='loso', average='macro'):
    print('kernel=', kernel, 'split=', split, 'average=', average)

    mean_acc_list = []
    real_acc_list = []
    f1_list = []
    mean_f1_list = []
    if kernel == 'rbf':
        for g in range(1, 10):
            gamma = g*0.1
            mean_acc, real_acc, f1, mean_f1 = avg_score(data, labels, sub_list, kernel=kernel,
                                                        gamma=gamma, split=split, average=average)
            mean_acc_list.append(mean_acc)
            real_acc_list.append(real_acc)
            f1_list.append(f1)
            mean_f1_list.append(mean_f1)
        print("best_real_acc_gamma:", np.argmax(real_acc_list)/10+0.1)
        print("best_real_acc:", max(real_acc_list))
        print("best_f1_gamma:", np.argmax(f1_list)/10+0.1)
        print("best_f1:", max(f1_list))
        print("best_mean_f1_gamma:", np.argmax(mean_f1_list)/10+0.1)
        print("best_mean_f1:", max(mean_f1_list))
    elif kernel == 'linear':
        for C in range(1, 10):
            mean_acc, real_acc, f1, mean_f1 = avg_score(data, labels, sub_list, kernel=kernel,
                                                        C=C, split=split, average=average)
            mean_acc_list.append(mean_acc)
            real_acc_list.append(real_acc)
            f1_list.append(f1)
            mean_f1_list.append(mean_f1)
        print("best_real_acc_C:", np.argmax(real_acc_list) + 1)
        print("best_real_acc:", max(real_acc_list))
        print("best_f1_C:", np.argmax(f1_list) + 1)
        print("best_f1:", max(f1_list))
        print("best_mean_f1_C:", np.argmax(mean_f1_list) + 1)
        print("best_mean_f1:", max(mean_f1_list))
    elif kernel == 'poly':
        for degree in range(1, 10):
            mean_acc, real_acc, f1, mean_f1 = avg_score(data, labels, sub_list, kernel=kernel,
                                                        degree=degree, split=split, average=average)
            mean_acc_list.append(mean_acc)
            real_acc_list.append(real_acc)
            f1_list.append(f1)
            mean_f1_list.append(mean_f1)
        print("best_real_acc_degree:", np.argmax(real_acc_list) + 1)
        print("best_real_acc:", max(real_acc_list))
        print("best_f1_degree:", np.argmax(f1_list) + 1)
        print("best_f1:", max(f1_list))
        print("best_mean_f1_degree:", np.argmax(mean_f1_list) + 1)
        print("best_mean_f1:", max(mean_f1_list))
    print('-----------------------------------------------')

