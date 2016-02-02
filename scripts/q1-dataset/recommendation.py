# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 22:29:14 2016

@author: Lee
"""

import numpy as np

def extract_txt_arr(filename):
  with open(filename, 'rb') as f:
    data = []
    for line in f:
        strSplit = line.split(' ')
        results = map(int, strSplit)
        data.append(results)
    return np.array(data)
    
alexRaw = extract_txt_arr('alex.txt')

R = extract_txt_arr('user-shows.txt')
R[499,0:100] = 0

m = R.shape[0] 
n = R.shape[1]

Pinverse_sqrt = np.zeros((m,m))
for x in range(0, m):
    Pinverse_sqrt[x,x] = 1/np.sqrt(np.sum(R[x,:]))

Su = np.dot(np.dot(np.dot(np.dot(Pinverse_sqrt, R),np.transpose(R)),Pinverse_sqrt),R)
Su_sp = Su[499,0:100]

temp = np.argpartition(-Su_sp, 5)
kUU = temp[:5]
print "5 highest indices"
print kUU

temp = np.partition(-Su_sp, 5)
kUU_score = -temp[:5] 
print "5 highest scores"
print kUU_score

#now using item to item

Qinverse_sqrt = np.zeros((n,n))

for x in range(0, n):
    Qinverse_sqrt[x,x] = 1/np.sqrt(np.sum(R[:,x]))

Si = np.dot(R,np.dot(np.dot(np.dot(Qinverse_sqrt, R),np.transpose(R)),Qinverse_sqrt))
Si_sp = Si[499,0:100]

temp = np.argpartition(-Si_sp, 5)
kII = temp[:5]
print "5 highest indices"
print kII

temp = np.partition(-Si_sp, 5)
kII_score = -temp[:5] 
print "5 highest scores"
print kII_score
