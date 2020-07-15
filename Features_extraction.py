#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import cv2

#uniform LBP-TOP
#length height width:
def get_LBP_TOP(img_seq, uniform_dict, x_radius = 1, y_radius = 1, t_radius = 4, xy_neighbor = 8, xt_neighbor = 8, yt_neighbor = 8):
    
    length, height, width = img_seq.shape[:3]
    bins = 59
    pi = 3.1415926
    hist = [[0 for j in range(bins)] for i in range(3)]
    hist = np.array(hist).astype('float64')
    x_border = x_radius
    y_border = y_radius
    t_border = t_radius
    
    for ti in range(t_border, length-t_border):
        for yi in range(y_border, height-y_border):
            for xi in range(x_border, width-x_border):
                center = img_seq[ti][yi][xi]
                
                #in XY plane
                basic_LBP = 0
                fea_bin = 0
                for p in range(0, xy_neighbor):
                    x = math.floor(xi + x_radius * math.cos((2 * pi * p) / xy_neighbor) + 0.5)
                    y = math.floor(yi - y_radius * math.sin((2 * pi * p) / xy_neighbor) + 0.5)
                    #print(x, y)
                    current = img_seq[ti][y][x]
                    if current >= center:
                        basic_LBP = basic_LBP + 2 ^ fea_bin
                    fea_bin = fea_bin + 1
                hist[0, uniform_dict[basic_LBP]] += 1
                
                #in XT plane
                basic_LBP = 0
                fea_bin = 0
                for p in range(0, xt_neighbor):
                    x = math.floor(xi + x_radius * math.cos((2 * pi * p) / xt_neighbor) + 0.5)
                    t = math.floor(ti + t_radius * math.sin((2 * pi * p) / xt_neighbor) + 0.5)
                    current = img_seq[t][yi][x]
                    if current >= center:
                        basic_LBP = basic_LBP + 2 ^ fea_bin
                    fea_bin = fea_bin + 1
                hist[1, uniform_dict[basic_LBP]] += 1
                
                #in YT plane
                basic_LBP = 0
                fea_bin = 0
                for p in range(0, yt_neighbor):
                    y = math.floor(yi - y_radius * math.sin((2 * pi * p) / yt_neighbor) + 0.5)
                    t = math.floor(ti + t_radius * math.cos((2 * pi * p) / yt_neighbor) + 0.5)
                    current = img_seq[t, y, xi]
                    if current >= center:
                        basic_LBP = basic_LBP + 2 ^ fea_bin
                    fea_bin = fea_bin + 1
                hist[2, uniform_dict[basic_LBP]] += 1
    
    #nomalize
    for i in range(3):
        hist[i] = hist[i]/sum(hist[i])
    
    return hist

def get_3DHOG(cell, xy_bins=8, xt_bins=12, yt_bins=12):
    pi = 3.1415926
    length, height, width = cell.shape[:3]
    cell = cell.astype('float32')
    xy_hist = np.array([0 for i in range(xy_bins)]).astype('float32')
    xt_hist = np.array([0 for i in range(xt_bins)]).astype('float32')
    yt_hist = np.array([0 for i in range(yt_bins)]).astype('float32')

    for ti in range(length):
        for yi in range(height):
            for xi in range(width):
                # x方向
                if xi == 0:
                    a = cell[ti, yi, xi + 1]
                elif xi == width - 1:
                    a = -cell[ti, yi, xi - 1]
                else:
                    a = -cell[ti, yi, xi - 1] + cell[ti, yi, xi + 1]
                # y方向
                if yi == 0:
                    b = cell[ti, yi + 1, xi]
                elif yi == height - 1:
                    b = -cell[ti, yi - 1, xi]
                else:
                    b = -cell[ti, yi - 1, xi] + cell[ti, yi + 1, xi]

                # T方向
                if ti == 0:
                    c = cell[ti + 1, yi, xi]
                elif ti == length - 1:
                    c = -cell[ti - 1, yi, xi]
                else:
                    c = -cell[ti - 1, yi, xi] + cell[ti + 1, yi, xi]

                # XY plane
                xy_val = math.sqrt(a * a + b * b)
                if xy_val != 0:
                    xy_theta = math.atan(a / (b + 0.01)) + pi / 2
                    xy_bin_width = pi / xy_bins
                    # print('a:', a, 'b:', b, 'c:', c)
                    # print('xy bin width:', xy_bin_width, 'xy theta:', xy_theta)
                    xy_hist[int((xy_theta // xy_bin_width) % xy_bins)] += xy_val

                # XT plane
                xt_val = math.sqrt(a * a + c * c)
                if xt_val != 0:
                    xt_theta = math.atan(a / (c + 0.01)) + pi / 2
                    xt_bin_width = pi / xt_bins
                    xt_index = xt_theta // xt_bin_width
                    xt_hist[int((xt_theta // xt_bin_width) % xt_bins)] += xt_val

                # YT plane
                yt_val = math.sqrt(b * b + c * c)
                if yt_val != 0:
                    yt_theta = math.atan(b / (c + 0.01)) + pi / 2
                    yt_bin_width = pi / yt_bins
                    yt_index = yt_theta // yt_bin_width
                    # print('yt bin width:', yt_bin_width, 'yt theta:', yt_theta)
                    yt_hist[int((yt_theta // yt_bin_width) % yt_bins)] += yt_val

    hist = []
    hist.append(xy_hist)
    hist.append(xt_hist)
    hist.append(yt_hist)
    for i in range(3):
        hist[i] /= sum(hist[i])

    return np.concatenate(hist)

def get_HOOF(img_seqs, bins=8):
    
    flow_data = []
    hist = []
    pi = 3.1415926
    for i in range(len(img_seqs)-1):
        img = img_seqs[i]
        next_img = img_seqs[i+1]
        flow = cv2.calcOpticalFlowFarneback(img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_data.append(flow)
    flow_data = np.array(flow_data)
    hist = np.array([0 for i in range(bins)]).astype('float32')
    for flow in flow_data:
        for arr in flow:
            for vec in arr:
                u, v = vec
                mag = math.sqrt(u*u+v*v)
                theta = math.atan(u/(v+0.01)) + pi/2
                bin_width = pi / bins
                hist[int((theta // bin_width) % bins)] += mag
    hist = np.array(hist)
    hist /= sum(hist)
    
    return hist

def seq_divide(image_seq, t_times=4, y_times=4, x_times=4):
    
    length, height, width = image_seq.shape[:3]
    sub_length, sub_height, sub_width = (length//t_times, height//y_times, width//x_times)
    new_seq = image_seq[0:t_times*sub_length, 0:y_times*sub_height, 0:x_times*sub_width]
    
    sub_blocks = []
    for ti in range(t_times):
        for yi in range(y_times):
            for xi in range(x_times):
                t_up = min((ti+1)*sub_length, length)
                y_up = min((yi+1)*sub_height, height)
                x_up = min((xi+1)*sub_width, width)
                sub_blocks.append(np.array(new_seq[ti*sub_length:t_up, yi*sub_height:y_up, xi*sub_width:x_up]))
                
    return np.array(sub_blocks)

#只有LBP-TOP需要uniform_dict
#feature: LBP-TOP、3DHOG、HOOF
def get_ep_features(ep, uniform_dict = None, feature='LBP-TOP', t_times=4, y_times=4, x_times=4,
                    x_radius = 1, y_radius = 1, t_radius = 4, xy_neighbor = 8, xt_neighbor = 8, yt_neighbor = 8,
                    xy_bins = 8, xt_bins = 12, yt_bins = 12,
                    bins=8):
    
    if len(ep.shape) == 4:
        gray_ep = []
        for image in ep:
            gray_ep.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        gray_ep = np.array(gray_ep)
    else:
        gray_ep = ep
    
    sub_blocks = seq_divide(gray_ep, t_times=t_times, y_times=y_times, x_times=x_times)
    hist = []
    for cell in sub_blocks:
        if feature == 'LBP-TOP':
            hist.append(get_LBP_TOP(cell, uniform_dict, x_radius=x_radius, y_radius=y_radius, t_radius=t_radius,
                                    xy_neighbor=xy_neighbor, xt_neighbor=xt_neighbor, yt_neighbor=yt_neighbor))
        elif feature == '3DHOG':
            hist.append(get_3DHOG(cell, xy_bins=xy_bins, xt_bins=xt_bins, yt_bins=yt_bins))
        elif feature == 'HOOF':
            hist.append(get_HOOF(cell, bins=bins))
            
    return np.array(hist).flatten()