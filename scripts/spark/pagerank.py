# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 21:01:09 2016

@author: Lee
"""

import sys
from pyspark import SparkContext

f_path = "hdfs://localhost/data/pgrank/links.txt"
f_local_path = "links.txt"

def computeContribs(neighbors, rank):
    for neighbor in neighbors: yield(neighbor, rank/len(neighbors))

sc = SparkContext()
  
links = sc.textFile(f_path)\
    .map(lambda line: line.split())\
    .map(lambda pages: (pages[0],pages[1]))\
    .distinct()\
    .groupByKey()\
    .persist()
    
numPages = links.count()
    
ranks = links.map(lambda (page,neighbors) : (page,1.0/numPages))

for x in xrange(10):
    contribs = links.join(ranks)\
        .flatMap(lambda (page,(neighbors, rank)) : \
            computeContribs(neighbors,rank))
    
    ranks = contribs\
        .reduceByKey(lambda v1, v2: v1+v2)\
        .map(lambda (page, contrib) : \
            (page, contrib * 0.85 + 0.15/numPages))
                    
for rank in ranks.collect(): 
    print rank        

sc.stop()