# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 22:29:14 2016

@author: Lee
"""

import numpy as np
import math

def extract_txt_arr(filename):
  with open(filename, 'rb') as f:
    data = []
    for line in f:
        strSplit = line.split(' ')
        results = map(int, strSplit)
        data.append(results)
    return np.array(data)
    
alexRaw = extract_txt_arr('alex.txt')
alexTest = np.copy(alexRaw)
alexTest[0, 0:100] = 0

R = extract_txt_arr('user-shows.txt')

m = R.shape[0] 
n = R.shape[1]

Pinverse_sqrt = np.zeros((m,m))
for x in range(0, m):
    Pinverse_sqrt[x,x] = 1/np.sqrt(np.sum(R[x,:]))

Lu = np.dot(np.dot(np.dot(np.dot(Pinverse_sqrt, R),np.transpose(R)),Pinverse_sqrt),R)


"""
Qinverse_sqrt = np.zeros((n,n))

for x in range(0, n):
    Q[x,x] = np.sum(R[:,x])
"""   
#Luu = np.linalg.inv(np.sqrt(P))*R*np.transpose(R)*np.linalg.inv(np.sqrt(P))*R
