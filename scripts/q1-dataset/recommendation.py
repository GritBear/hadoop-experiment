# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 22:29:14 2016

@author: Lee
"""

import numpy as np

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
        strSplit = line.split(' ')
        results = map(int, strSplit)
        data.append(results)
    return np.array(data)

ind2name = extract_txt_vec('shows.txt')

alexRaw = extract_txt_arr('alex.txt')

R = extract_txt_arr('user-shows.txt')
R[499,:100] = 0

m = R.shape[0] 
n = R.shape[1]

Pinverse_sqrt = np.zeros((m,m))
for x in range(0, m):
    Pinverse_sqrt[x,x] = 1/np.sqrt(np.sum(R[x,:]))

Su = np.dot(np.dot(np.dot(np.dot(Pinverse_sqrt, R),np.transpose(R)),Pinverse_sqrt),R)
Su_sp = Su[499,0:100]

kUU = [i[0] for i in sorted(enumerate(-Su_sp), key=lambda x:x[1])]

print "5 highest UU indices"
print kUU[:5]

kUU_score = Su_sp[kUU] 
print "5 highest UU scores"
print kUU_score[:5]

print "5 highest UU shows"
for x in range(0,5):
    print ind2name[kUU[x]]

print "======================================="
print ""

#now using item to item
Qinverse_sqrt = np.zeros((n,n))

for x in range(0, n):
    Qinverse_sqrt[x,x] = 1/np.sqrt(np.sum(R[:,x]))

Si = np.dot(R,np.dot(np.dot(np.dot(Qinverse_sqrt, np.transpose(R)), R),Qinverse_sqrt))
Si_sp = Si[499,:100]

kII = [i[0] for i in sorted(enumerate(-Si_sp), key=lambda x:x[1])]

print "5 highest II indices"
print kII[:5]

kII_score = Si_sp[kII] 
print "5 highest II scores"
print kII_score[:5]

print "5 highest II shows"
for x in range(0,5):
    print ind2name[kII[x]]
    
print "======================================="
print ""
    
#Plot the accuracy curve
alexSp = alexRaw[0,:100];
numPositive = np.sum(alexSp)

UU_rate = []
II_rate = []
cnt_UU = 0
cnt_II = 0
for x in range(0,20):
    indUU = kUU[x]
    indII = kII[x]
    if alexSp[indUU] == 1:
        cnt_UU = cnt_UU + 1
    if alexSp[indII] == 1:
        cnt_II = cnt_II + 1
    
    UU_rate.append(float(cnt_UU)/float(numPositive))
    II_rate.append(float(cnt_II)/float(numPositive))

print "UU rate"
print  UU_rate
print "II rate"
print  II_rate

print "======================================="
print ""