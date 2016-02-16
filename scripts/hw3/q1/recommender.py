# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:44:17 2016

@author: Lee
"""

import numpy as np
import math
import matplotlib.pyplot as plt

TrainFileName = 'ratings.train.txt'

TestFileName = 'ratings.val.txt'

def countFileMax(filename):
    with open(filename, 'rb') as f:
        u = 0
        i = 0
        for line in f:
            strSplit = line.split('\t')
            results = map(int, strSplit)
            u = max(u, results[0])
            i = max(i, results[1])
        return u,i

MaxUser, MaxItem = countFileMax(TrainFileName)

print MaxUser
print MaxItem

k = 20

# Q is m by k, maxItem by k
# P is n by k, maxUser by k

# initialize P and Q
P = np.random.rand(MaxUser,k) * np.sqrt(5.0/k)
Q = np.random.rand(MaxItem, k) * np.sqrt(5.0/k)

#print P
#print Q

def OneIteration(filename, learningRate, P, Q, regularize):
    Error = regularize * (pow(np.linalg.norm(P),2) + pow(np.linalg.norm(Q),2))    
    with open(filename, 'rb') as f:
        for line in f:
            strSplit = line.split('\t')
            results = map(int, strSplit)
            userId = results[0]
            itemId = results[1]
            rating = results[2]        
            
            # store old values
            qi = np.copy(Q[itemId-1,:])
            pu = np.copy(P[userId-1,:])    
            
            Error = Error + pow((rating - np.dot(qi,np.transpose(pu))),2) 

            if(math.isnan(Error)):
                print 'iteration debug'
                print Error
                print(pow((rating - np.dot(qi,np.transpose(pu))),2))
                print((np.dot(qi,np.transpose(qi)) + np.dot(pu,np.transpose(pu))))     
                input('Press <ENTER> to continue')
            
            # perform update            
            err_iu = 2*(rating - np.dot(qi,np.transpose(pu)))
            
            # perform update
            Q[itemId-1,:] = Q[itemId-1,:] + learningRate * (err_iu * pu - regularize * qi)
            P[userId-1,:] = P[userId-1,:] + learningRate * (err_iu * qi - regularize * pu)            
        
        return Error, P, Q


maxIteration = 40
regularize = 0.2
learningRate = 0.01
ErrorArr = []    

for x in range(0, maxIteration + 1):
    Error, P, Q = OneIteration(TrainFileName, learningRate, P, Q, regularize)
    print x
    print Error
    ErrorArr.append(Error)

# plot ErrorArr
plt.plot(ErrorArr)
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.show()


