# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 14:42:46 2016

@author: Lee
"""
import numpy as np

inputCount = "counts.txt"
inputStream = "words_stream.txt"

numHash = 5
numBucket = 10000
P = 123457

hashParam = [[3,1561],  \
            [17,277],   \
            [38,394],   \
            [61,13],    \
            [78,246]]

def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a*y + b) % p
    return hash_val % n_buckets
    
def recallF(hashTable, hashParam, numHash, eventId):
    t_count = -1
    for h in range(numHash):
        a = hashParam[h][0]
        b = hashParam[h][1]
        bucketId = hash_fun(a,b,P,numBucket,eventId)
        if(t_count == -1):
            t_count = hashTable[h][bucketId]
        else:
            t_count = min(t_count, hashTable[h][bucketId])
    return t_count

def extract_txt_arr(filename):
  with open(filename, 'rb') as f:
    data = []
    for line in f:
        strSplit = line.split('\t')
        results = map(int, strSplit)
        data.append(results)
    return np.array(data)
    
rightCount = extract_txt_arr(inputCount)
n_distinct = rightCount.shape[0]

# initialize memory
hashTable = np.zeros([numHash, numBucket],dtype=np.int)
t = 0

# run Stream processing
with open(inputStream, 'rb') as f:
    for line in f:
        eventId = int(line.rstrip()) #i
        
        for h in range(numHash):
            a = hashParam[h][0]
            b = hashParam[h][1]
            bucketId = hash_fun(a,b,P,numBucket,eventId)
            hashTable[h][bucketId] = hashTable[h][bucketId] + 1
        
        t = t+1

#run test
plotX = []
plotY = []

for x in range(n_distinct):
    eventId = rightCount[x,0]
    F_right = rightCount[x,1]
    x_val = rightCount[x,1] / float(t)
    F_estimate = recallF(hashTable, hashParam, numHash, eventId)
    y_val = (F_estimate - F_right) / float(F_right)
    
    plotX.append(x_val)
    plotY.append(y_val)
    
import matplotlib.pyplot as plt
plt.plot(plotX, plotY, '.')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Relative Error')
plt.xlabel('Nomalized Frequency')
plt.title('Nomalized Frequency vs Relative Error (hw4q4)')
plt.show()
