# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:13:00 2016

@author: Lee
"""

import numpy as np

FileName = 'graph.txt'

M_raw = np.zeros([100,100])
M = np.zeros([100,100])

def extract_txt_arr(filename, M_raw):
  with open(filename, 'rb') as f:
    for line in f:
        strSplit = line.split('\t')
        results = map(int, strSplit)
        srcId = results[0] - 1
        targetId = results[1] - 1       
        M_raw[targetId, srcId] = 1
    return M_raw
    
M_raw = extract_txt_arr(FileName, M_raw)

# normalize M_raw by column
for x in range(0,100):
    col = M_raw[:,x]
    colSum = np.sum(col)
    M[:,x] = col/colSum
    
# init r
r = np.ones([100,1]) / 100
beta = 0.8
constant1 = np.ones([100,1]) * (1 - beta) / 100

for x in range(0, 40):
    r = constant1 + beta * np.dot(M,r)
    
r_sort = sorted(range(len(r)), key=lambda i: r[i])

indexOffset = np.ones([5]).astype(int)

# top 5
print 'Top 5 PageRank'
print r_sort[-5:] + indexOffset # low to high
print 'Bottom 5 PageRank'
print r_sort[:5] + indexOffset # low to high


# hits 
L = np.copy(np.transpose(M_raw))
h = np.ones([100,1])
a = np.ones([100,1])
constantL = 1
constantA = 1

for x in range(0,40):
    a = np.dot(np.transpose(L),h)
    h = np.dot(L, a)
    
h_sort = sorted(range(len(r)), key=lambda i: h[i])
a_sort = sorted(range(len(r)), key=lambda i: a[i])
    
# top 5
print 'Top 5 hubbiness'
print h_sort[-5:] + indexOffset # low to high
print 'Bottom 5 hubbiness'
print h_sort[:5] + indexOffset # low to high

# top 5
print 'Top 5 authority'
print a_sort[-5:] + indexOffset # low to high
print 'Bottom 5 authority'
print a_sort[:5] + indexOffset # low to high    


