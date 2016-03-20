# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 23:30:42 2016

SVM script

@author: Lee
"""
import time
import numpy as np
import random

#load the data
TrainingFile = "features.txt"
LabelFile = "target.txt"

def extract_txt_vec(filename):
  with open(filename, 'rb') as f:
    data = []
    for line in f:
        data.append(line.rstrip())
    return np.array(data).astype(int)

def extract_txt_arr(filename):
  with open(filename, 'rb') as f:
    data = []
    for line in f:
        strSplit = line.split(',')
        results = map(int, strSplit)
        data.append(results)
    return np.array(data)
    
X_all = extract_txt_arr(TrainingFile)
Y_all = extract_txt_vec(LabelFile)
Y_all = Y_all.reshape(Y_all.shape[0],1)

def calcCost(X,Y,W,b,C): 
    
    # calc prediction
    pred = np.dot(X,W) + b
    ypred = np.multiply(Y,pred)
    good_prediction_indices = ypred >= 1   
    L = 1.0 - ypred
    L[good_prediction_indices] = 0
    
    cost = W.transpose().dot(W) * 0.5  + C * np.sum(L) 
    return cost

def calcDelta(X,Y,W,b,C):
    isSGD = (len(X.shape) > 1)    
    
    # calc prediction
    pred = np.dot(X,W) + b
    ypred = np.multiply(Y,pred)
    good_prediction_indices = ypred >= 1    
    
    Lmark = -np.ones(Y.shape)
    Lmark[good_prediction_indices] = 0
    
    yx = np.multiply(Y, X)
    negative_yx = np.multiply(Lmark, yx) #conditional filter
    
    Lw = 0.0
    if(isSGD):
        Lw = C * np.sum(negative_yx,0).reshape(W.shape)
    else:
        Lw = C * negative_yx.reshape(W.shape)
    
    deltaW = W + Lw
    
    negative_y = np.multiply(Y, Lmark)
    deltab = C * np.sum(np.sum(negative_y))     
    
    return deltaW, deltab
    
#def calcCostDeltaSDG(X,Y,W,b,C):
#    # calc prediction
#    ypred = Y * (np.dot(X,W) + b)
#    
#    if(ypred >= 1):
#        # it is a correct sample
#        deltaW = W
#        deltab = 0
#        cost = np.dot(np.transpose(W),W) * 0.5
#    else:
#        # it is wrong
#        cost = np.dot(np.transpose(W),W) * 0.5  + C * (1-ypred)
#        yx = X * Y
#        Lw = -C * yx.reshape(W.shape)
#        deltaW = W + Lw
#        deltab = -C * Y
#    
#    return cost.reshape(1)[0], deltaW, deltab

def batchDeltaCost(f_prev, f_cur):
    return abs(f_prev - f_cur) * 100 / f_prev
    
def SGDDeltaCost(f_prev, f_cur, fdelta_prev):
    return 0.5 * fdelta_prev + 0.5 * batchDeltaCost(f_prev, f_cur)



def runBatch(X_all, Y_all):
    C = 100
    b = 0
    W = np.zeros([X_all.shape[1],1])
    
    costArr = []
    learnRate = 0.0000003
    stopCriteria = 0.25
    f_delta = 1000000 
    
    k = 0
    tic = time.clock()
    while(f_delta > stopCriteria and k < 2000):
        deltaW, deltab = calcDelta(X_all, Y_all, W, b, C)
        cost = np.sum(calcCost(X_all, Y_all, W, b, C))
        #update params
        W = W - learnRate * deltaW
        b = b - learnRate * deltab
        costArr.append(cost)
        
        if(k > 0):
            f_delta = batchDeltaCost(costArr[k-1], cost)
        
        k = k + 1
    
    toc = time.clock()

    return costArr, toc - tic, k
    
def runSDG(X_all, Y_all):
    C = 100
    b = 0
    W = np.zeros([X_all.shape[1],1])
    
    n = X_all.shape[0]
    
    order = range(0,n)    
    random.shuffle(order)    
    
    costArr = []
    costDeltaArr = []
    learnRate = 0.0001
    stopCriteria = 0.001
    f_delta = 90000.0
    
    k = 0
    i = 0
    tic = time.clock()
    while(f_delta > stopCriteria and k < 6000):
        X = X_all[order[i],:]
        Y = Y_all[order[i]]      
        
        deltaW, deltab = calcDelta(X, Y, W, b, C)
        cost = np.sum(calcCost(X_all, Y_all, W, b, C))
        #update params
        W = W - learnRate * deltaW
        b = b - learnRate * deltab
        costArr.append(cost)
        
        if(k > 0):
            if(f_delta == 90000.0):
                f_delta = 0.0 # start from 0
                
            f_delta = 0.5 * f_delta + 0.5 * batchDeltaCost(costArr[k-1], cost)
            costDeltaArr.append(f_delta)
        
        k = k + 1
        i = (i + 1) % n
        
    toc = time.clock()

    return costArr, toc - tic, k, costDeltaArr
    
    
def runMiniBatch(X_all, Y_all):
    C = 100
    b = 0
    W = np.zeros([X_all.shape[1],1])
    
    n = X_all.shape[0]
    batchSize = 20
    
    maxBatch = (n / 20) + 1
    
    order = range(0,n)    
    random.shuffle(order)    
    
    costArr = []
    costDeltaArr = []
    learnRate = 0.00001
    stopCriteria = 0.01
    f_delta = 90000.0
    
    k = 0
    l = 0
    tic = time.clock()
    while(f_delta > stopCriteria and k < 6000):
        start = l *  batchSize      
        end = min(n,(l + 1) * batchSize)       
        selection = order[start:end]
        X = X_all[selection,:]
        Y = Y_all[selection]    
        deltaW, deltab = calcDelta(X, Y, W, b, C)
        cost = np.sum(calcCost(X_all, Y_all, W, b, C))
        #update params
        W = W - learnRate * deltaW
        b = b - learnRate * deltab
        costArr.append(cost)
        
        if(k > 0):
            if(f_delta == 90000.0):
                f_delta = 0.0
                
            f_delta = 0.5 * f_delta + 0.5 * batchDeltaCost(costArr[k-1], cost)
            costDeltaArr.append(f_delta)
        
        k = k + 1
        l = (l + 1) % maxBatch
        
    toc = time.clock()

    return costArr, toc - tic, k, costDeltaArr
    
batchCostArr, batchTime, batchIteration = runBatch(np.copy(X_all), np.copy(Y_all))
SGDCostArr, SGDTime, SGDIteration, SGDcostDeltaArr = runSDG(np.copy(X_all), np.copy(Y_all))
mBatchCostArr, mBatchTime, mBatchIteration, mBatchcostDeltaArr = runMiniBatch(np.copy(X_all), np.copy(Y_all))

batchTime
SGDTime
mBatchTime

batchIteration
SGDIteration
mBatchIteration

import matplotlib.pyplot as plt
plt.plot(batchCostArr)
plt.plot(SGDCostArr)
plt.plot(mBatchCostArr)

plt.legend(['Batch', 'SGD', 'MiniBatch'], loc='upper right')

plt.ylabel('Error')
plt.xlabel('Iteration')
plt.title('Cost vs Iteration')
plt.show()