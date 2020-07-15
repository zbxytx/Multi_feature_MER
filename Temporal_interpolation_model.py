#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import numpy.matlib as npml
import numpy.matlib as nml

class TIM:
    def __init__(self):
        pass
    
    def getPathWeightMatrix(self, N):
        G = np.zeros((N,N))
        G = np.asmatrix(G)
        for i in range(N-1):
            G[i,i+1] = 1
            G[i+1,i] = 1
        return G

    def getLapacian(self, G):
        N, M = G.shape
        D= np.zeros((N,N))
        sum_G = np.sum(G,axis = 0)
        for i in range(N):
            D[i,i] = sum_G[0,i]

        L = D - G
        return L

    #X = (n, m) n=向量化图像的维度 m=图像原始帧数
    def TrainPGM(self, X):
        (n,m) = X.shape
        rank_X = np.linalg.matrix_rank(X)
        if (rank_X < m):
            print(" ERROR : Invalid Input ")
            return -1

        mu = np.mean(X, axis =1)
        X2 = X - nml.repmat(mu,1,m)

        u,s,vh = np.linalg.svd(X2, full_matrices = False)

        u = -1*u
        #sm:奇异值分解中的对角矩阵
        sm = np.zeros((m,m))
        sm = np.asmatrix(sm)

        for i in range(m):
            sm[i,i] = s[i]

        V = vh.T.conj() 
        # S(N,:) = []
        sm = np.delete(sm,m-1,axis=1)
        sm = np.delete(sm,m-1,axis=0)

        V = np.delete(V,m-1,axis=1)

        u = np.delete(u,m-1,axis=1)
        u = -1*u

        Q = np.dot(sm , np.transpose(V))

        G = self.getPathWeightMatrix(m)
        L = self.getLapacian(G)

        #V0:L的特征向量矩阵
        ew,ev = np.linalg.eigh(L,UPLO='L')
        V0 = np.delete(ev,0,axis = 1)

        p1 = np.dot(Q,np.transpose(Q))
        p2 = np.dot(Q,V0)

        W = np.dot(np.linalg.inv(p1),p2)

        mat = np.zeros((m,1))

        # m(j) = Q(:,1)'*W(:,j)/sin(1/N*j*pi+pi*(N-j)/(2*N));
        pi = 3.1415927
        for i in range(1,m):
            val1 = np.dot( np.transpose(Q[:,0]), W[:,i-1] )
            val2 = np.sin( (1.0/m)* i * np.pi + np.pi*(m-i)/(2*m))
            mat[i] = np.asscalar(val1) / val2

        #mat: 对角矩阵m，用一阶向量表示
        #m:初始帧数
        #mu:平均向量
        #u:奇异值分解获得的矩阵U
        #W:(Q*Q^T)^-1*Q*V0
        model = {'W':W , 'U': u, 'mu': mu, 'num': m, 'mat':mat }
        return model
    
    def synPGM(self, model, pos):
        PI = 3.14159
        n = model['num']
        pos2 = pos*(1- 1.0 /float(n)) + 1.0 / float(n)
        # synthesis

        n_model_U = model['U'].shape[0]
        n_pos = pos2.shape[0]
        X = np.zeros((n_model_U,n_pos),dtype=float)

        ndim = model['W'].shape[0]

        modelU = model['U']
        modelW = model['W']
        modelM = model['mat']
        modelMu = model['mu']

        modelM = modelM[1:n,0]
        for i in range (n_pos):
            v = np.zeros((ndim,1),dtype=float)
            for k in range (ndim):
                v[k] = np.sin(pos[i]*(k+1)* PI + PI*(n- (k+1)) / (2*n))
            X[:,i] = np.array(modelU * (np.linalg.inv(modelW)*(v*modelM.reshape(-1, 1))) + modelMu).flatten()
        return X
    
    def run(self, image_seq, target_length):
        seq_length, height, width, channels = image_seq.shape[:4]
        vector_length = height*width*channels
        img_vector = np.zeros((vector_length, seq_length))
        for i, img in enumerate(image_seq):
            img_vector[:, i] = np.reshape(img, vector_length)

        img_vector =  np.asmatrix(img_vector)
        model = self.TrainPGM(img_vector)

        pos_list = []
        st= 1.0 / target_length
        curr = st
        for _ in range(target_length):
            pos_list.append(curr)
            curr = curr + st
        pos = np.asarray(pos_list)

        X_new = self.synPGM(model, pos)
        new_seq = []
        for i in range(X_new.shape[1]):
            new_seq.append(np.reshape(X_new[:, i], (height, width, channels)))

        final = np.array(new_seq)

        #防止结果超出[0, 255]
        final[final<18] = (18-final[final<18])/(18-np.min(final))*18
        final[final > 238] = 238+(final[final > 238]-238)/(np.max(final)-238)*17

        return final.astype('uint8')




