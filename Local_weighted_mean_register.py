#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import dlib
import cv2
import imutils
from scipy import optimize

#1. 选择标准脸(第一个EP的第一帧)
#2. 将标准脸放正（alignment）
#3. 对每个序列，对第一帧alignment，随后计算第一帧到标准脸的LWM映射
#4. 对序列中的所有图像应用映射
#input: standard_face, image_seq
#params: width, height, offset, itype:'RGB' or 'gray'

from collections import OrderedDict

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
 ("right_eye", (2, 3)),
 ("left_eye", (0, 1)),
 ("nose", (4)),
])

def shape_to_np(shape, dtype='int'):
    # 创建68*2用于存放坐标
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def cp_resize(shape, size, resize, xmin, ymin):
    
    x = np.array([item[0] for item in shape])
    y = np.array([item[1] for item in shape])
    
    x = x-xmin
    y = y-ymin
    for i in range(len(x)):
        x[i] = round(float(x[i]/size[1]*resize[1]))
        y[i] = round(float(y[i]/size[0]*resize[0]))
        
    return np.array([[x[i], y[i]] for i in range(len(x))])

class FaceAligner:
    def __init__(self, predictor_path, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor_path = predictor_path
        self.predictor = dlib.shape_predictor(predictor_path)    # 获取人脸检测器
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
 
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
 
    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
 
        # simple hack ;)
        if (len(shape) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
 
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
 
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
 
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
 
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
 
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
 
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)
 
        # return the aligned face
        return output
    
    #对图像序列进行align
    #根据第一张图像的rect，计算所有图像的alignment
    def seq_align(self, image_seq, itype='RGB'):
        detector = dlib.get_frontal_face_detector()
        predictor = self.predictor
    
        image = image_seq[0]
        if itype == 'RGB':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        new_face_seq = []
        rects = detector(gray, 1)
        for rect in rects:
            for face in image_seq:
                (x, y, w, h) = rect_to_bb(rect)
                # 调用 align 函数对图像中的指定人脸进行处理
                face_aligned = self.align(face, gray, rect)
                new_face_seq.append(face_aligned)
    
        return np.array(new_face_seq)
    
    def run(self, image_seq, itype='RGB'):
        return self.seq_align(image_seq, itype=itype)

class LWMRegister:
    def __init__(self, standard_face, predictor_path, width=192, height=192, offset=24):
        self.width = width
        self.height = height
        self.offset = offset

        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = predictor_path
        self.predictor = dlib.shape_predictor(predictor_path)
        self.fa = FaceAligner(predictor_path = self.predictor_path, desiredFaceWidth=width+2*offset, desiredFaceHeight=height+2*offset)
        
        aligned_face = self.fa.seq_align(np.array([standard_face]))[0]
        dets = self.detector(aligned_face, 1)
        standard_shape = self.predictor(aligned_face, dets[0])
        standard_xmin = min([pt.x for pt in standard_shape.parts()])
        standard_xmax = max([pt.x for pt in standard_shape.parts()])+1
        standard_ymin = min([pt.y for pt in standard_shape.parts()])
        standard_ymax = max([pt.y for pt in standard_shape.parts()])+1
        
        self.standard_face = cv2.resize(aligned_face[standard_ymin:standard_ymax, standard_xmin:standard_xmax],
                                   (height, width),interpolation=cv2.INTER_CUBIC)
        ori_size = aligned_face[standard_ymin:standard_ymax, standard_xmin:standard_xmax].shape
        self.standard_shape = cp_resize(shape_to_np(standard_shape), ori_size, (height, width), standard_xmin, standard_ymin)
    
    #根据map计算新的图像，如果map中有空白像素，用周围点的map补充
    def remap_complete(self, image, map_x, map_y, xmin, xmax, ymin, ymax, channels = 3):
        #map_x[i, j] = k: 目标图像(i, j)处的原图像横坐标为k
        offset = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        empty_list = []
        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                if map_x[i, j] == -1:#map_x为空则map_y也为空
                    point = 0
                    result_x = 0
                    result_y = 0
                    for x_offset, y_offset in offset:
                        y = y_offset+i
                        x = x_offset+j
                        if map_x[y, x] != -1:
                            point += 1
                            result_x += map_x[y, x]
                            result_y += map_y[y, x]
                    if point != 0:
                        map_x[i, j] = result_x/point
                        map_y[i, j] = result_y/point
                
                    if map_x[i, j] == -1:
                        empty_list.append((i, j))
    
        while len(empty_list) != 0:
            temp_list = empty_list
            for i, j in temp_list:
                point = 0
                result_x = 0
                result_y = 0
                for x_offset, y_offset in offset:
                    y = y_offset+i
                    x = x_offset+j
                    if map_x[y, x] != -1:
                        point += 1
                        result_x += map_x[y, x]
                        result_y += map_y[y, x]
                if point != 0:
                    map_x[i, j] = result_x/point
                    map_y[i, j] = result_y/point

                if map_x[i, j] != -1:
                    empty_list.remove((i, j))
    
        new_image = np.zeros((ymax-ymin, xmax-xmin, channels))
        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                if map_x[i, j] == -1 and map_y[i, j] == -1:
                    continue
                a = math.floor(map_x[i, j])
                b = math.ceil(map_x[i, j])
                c = math.floor(map_y[i, j])
                d = math.ceil(map_y[i, j])
                x = map_x[i, j]
                y = map_y[i, j]

                if a == b:
                    txy1 = image[c, a]
                    txy2 = image[d, a]
                else:
                    t00 = image[c, a]
                    t01 = image[c, b]
                    t10 = image[d, a]
                    t11 = image[d, b]
                    txy1 = ((x-a)*t00 + (b-x)*t01)/(b-a)
                    txy2 = ((x-a)*t10 + (b-x)*t11)/(b-a)

                if c == d:
                    txy = txy1
                else:
                    txy = ((y-c)*txy1+(d-y)*txy2)/(d-c)
                new_image[i-ymin, j-xmin] = np.round(txy)

        return new_image.astype('uint8')
    
    #获得映射函数
    def get_map(self, standard_face, standard_shape, aligned_seq, detector, predictor, n = 6, width = 192, height = 192, offset = 24, amplify = 1.2):

        face = aligned_seq[0]
        #standard x y target x y
        sx = np.array([item[0] for item in standard_shape])
        sy = np.array([item[1] for item in standard_shape])

        dets = detector(face, 1)
        shape = shape_to_np(predictor(face, dets[0]))

        xmin = min([item[0] for item in shape])
        xmax = max([item[0] for item in shape])+1
        ymin = min([item[1] for item in shape])
        ymax = max([item[1] for item in shape])+1
        face = face[int(ymin-round(offset/height*(ymax-ymin))):int(ymax+round(offset/height*(ymax-ymin))),
                    int(xmin-round(offset/width*(xmax-xmin))):int(xmax+round(offset/width*(xmax-xmin)))]
        face_size = face.shape
        face = cv2.resize(face,(int(amplify*width+2*amplify*offset), int(amplify*height+2*amplify*offset)),
                          interpolation=cv2.INTER_CUBIC)
        shape = cp_resize(shape, face_size, (int(amplify*width+2*amplify*offset), int(amplify*height+2*amplify*offset)), 
                            xmin-round(offset/width*(xmax-xmin)), ymin-round(offset/height*(ymax-ymin)))

        tx = np.array([item[0] for item in shape])
        ty = np.array([item[1] for item in shape])

        length = len(sx)
        coeffX = np.zeros((length, 6))
        coeffY = np.zeros((length, 6))
        radii = np.zeros((length)) #每个cp距离第N近的点

        fitfunc = lambda p, x: np.dot(x,p)
        errfunc = lambda p, x, y: fitfunc(p, x) - y

        #find n nearest point
        distance = np.zeros((length, length))
        for i in range(length):
            dist = np.sqrt( (tx-tx[i])**2 + (ty-ty[i])**2 )
            index = dist.argsort()
            dist_sorted = dist[index]
            radii[i] = dist_sorted[n-1]
            neighbors_index = index[:n].sort()

            #第i个cp的n个临近cp
            sxi = sx[neighbors_index]
            syi = sy[neighbors_index]
            txi = tx[neighbors_index]
            tyi = ty[neighbors_index]

            X = (np.row_stack((np.ones((1, length)), txi, tyi, txi*tyi, txi**2, tyi**2))).T
            para = np.zeros(6)

            #lsq获取拟合多项式
            coeffXi,success = optimize.leastsq(errfunc, para, args=(X,sxi.ravel()))
            coeffYi,success = optimize.leastsq(errfunc, para, args=(X,syi.ravel()))
            coeffX[i] = coeffXi
            coeffY[i] = coeffYi

        #get map
        W = lambda r : 1-3*r*r+2*r*r*r if r <= 1 or r >= 0 else 0
        map_x = np.full((height+2*offset, width+2*offset), -1)
        map_y = np.full((height+2*offset, width+2*offset), -1)
        for i in range(int(amplify*height+2*amplify*offset)):
            for j in range(int(amplify*width+2*amplify*offset)): #对每个像素

                wk = np.sqrt((ty-i)**2 + (tx-j)**2)/radii
                wk[wk < 0] = 0
                x_sk = np.dot(coeffX, np.array([1, j, i, i*j, j**2, i**2]).T)
                y_sk = np.dot(coeffY, np.array([1, j, i, i*j, j**2, i**2]).T)
                x_up = np.dot(wk, x_sk).sum()
                y_up = np.dot(wk, y_sk).sum()
                bottom = wk.sum()

                if bottom == 0:
                    print("bottom = 0:", i, j)
                    continue

                x_new = round(float(x_up/bottom)) + offset
                y_new = round(float(y_up/bottom)) + offset

                if x_new < 0 or x_new >= map_x.shape[1] or y_new < 0 or y_new >= map_x.shape[0]:
                    #print("out of range", i, j, "new y, x", y_new, x_new)
                    continue
                else:
                    map_x[y_new, x_new] = j
                    map_y[y_new, x_new] = i

        return map_x, map_y
    
    def image_register(self, standard_face, standard_shape, face_seq, detector, predictor,
                       width = 192, height = 192, offset = 24, n = 6, amplify = 1.2):

        #aligned_seq = seq_align(face_seq, width=width, height=height, offset=offset)
        aligned_seq = face_seq
        face = aligned_seq[0]
        #standard x y target x y
        sx = np.array([item[0] for item in standard_shape])
        sy = np.array([item[1] for item in standard_shape])

        dets = detector(face, 1)
        shape = shape_to_np(predictor(face, dets[0]))

        xmin = min([item[0] for item in shape])
        xmax = max([item[0] for item in shape])+1
        ymin = min([item[1] for item in shape])
        ymax = max([item[1] for item in shape])+1
        face = face[int(ymin-round(offset/height*(ymax-ymin))):int(ymax+round(offset/height*(ymax-ymin))),
                    int(xmin-round(offset/width*(xmax-xmin))):int(xmax+round(offset/width*(xmax-xmin)))]
        face_size = face.shape
        face = cv2.resize(face,(int(amplify*width+2*amplify*offset), int(amplify*height+2*amplify*offset)),
                          interpolation=cv2.INTER_CUBIC)
        shape = cp_resize(shape, face_size, (int(amplify*width+2*amplify*offset), int(amplify*height+2*amplify*offset)), 
                            xmin-round(offset/width*(xmax-xmin)), ymin-round(offset/height*(ymax-ymin)))

        tx = np.array([item[0] for item in shape])
        ty = np.array([item[1] for item in shape])

        map_x, map_y = self.get_map(standard_face, standard_shape, aligned_seq, detector=detector, predictor=predictor,
                               n = n, width = width, height = height, offset = offset, amplify = amplify)

        temp_amp = amplify
        while (map_x < 0).any() and temp_amp < 3:
            temp_amp += 0.1
            new_map_x, new_map_y = self.get_map(standard_face, standard_shape, aligned_seq, detector=detector, predictor=predictor,
                                   n = n, width = width, height = height, offset = offset, amplify = temp_amp)
            map_x[map_x<0] = new_map_x[map_x<0]/temp_amp*amplify
            map_y[map_y<0] = new_map_y[map_y<0]/temp_amp*amplify

        new_face_seq = []
        for image in aligned_seq:

            coord_ymin = max(int(ymin-round(offset/height*(ymax-ymin))), 0)
            coord_ymax = int(ymax+round(offset/height*(ymax-ymin)))
            coord_xmin = max(int(xmin-round(offset/width*(xmax-xmin))), 0)
            coord_xmax = int(xmax+round(offset/width*(xmax-xmin)))
            temp_image = image[coord_ymin:coord_ymax, coord_xmin:coord_xmax]
            temp_image = cv2.resize(temp_image,(int(amplify*width+2*amplify*offset), int(amplify*height+2*amplify*offset)), interpolation=cv2.INTER_CUBIC)

            new_image = self.remap_complete(temp_image, map_x, map_y, offset, width+offset+1, offset, height+offset+1)
            new_face_seq.append(new_image)

        return np.array(new_face_seq)
    
    def run(self, face_seq, n = 6, amplify = 1.2, aligned=False):
        if aligned == True:
            return self.image_register(self.standard_face, self.standard_shape, face_seq, self.detector, self.predictor,
                                  width = self.width, height = self.height, offset = self.offset, n = n, amplify = amplify)
        elif aligned == False:
            result = self.fa.seq_align(face_seq)
            return self.image_register(self.standard_face, self.standard_shape, result, self.detector, self.predictor,
                                  width = self.width, height = self.height, offset = self.offset, n = n, amplify = amplify)
