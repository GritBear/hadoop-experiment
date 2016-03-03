# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 23:30:42 2016

SVM script

@author: Lee
"""

import numpy as np

#load the data
TrainingFile = "features.txt"
LabelFile = "target.txt"

def extract_txt_vec(filename):
  with open(filename, 'rb') as f:
    data = []
    for line in f:
        data.append(line.rstrip())
    return data

def extract_txt_arr(filename):
  with open(filename, 'rb') as f:
    data = []
    for line in f:
        strSplit = line.split(',')
        results = map(int, strSplit)
        data.append(results)
    return np.array(data)
    
X = extract_txt_arr(TrainingFile)
Y = extract_txt_arr(LabelFile)

b = 0
W = np.zeros([X.shape[1],1])
C = 100

def calcCostDelta(X,Y,W,b):
    cost = 0
    #calc error
    cost = cost + np.dot(np.transpose(W),W) * 0.5    
    
    
    return 0