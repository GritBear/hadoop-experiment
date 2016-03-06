# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 23:30:42 2016

SVM script

@author: Lee
"""

import sys
sys.modules[__name__].__dict__.clear()

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

from sklearn import svm
clf = svm.SVC()
clf.fit(X_all, Y_all)
W = clf.support_vectors_
print clf.support_vectors_