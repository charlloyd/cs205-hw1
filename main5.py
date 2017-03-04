import os
import numpy as np
import hw14
import hw14opt
import time
import csv
import math
import random
from scipy.linalg.blas import dgemm

n=4

sizes = [2**6, 2**10]#, 2**12]
iter = range(len(sizes))
matlist = [np.ones((sizes[i],sizes[i]),dtype=np.float64) for i in iter]

i=0
X = matlist[i]

outmat = np.zeros((sizes[i],sizes[i]))
row =  int(round(np.floor((np.sqrt(16*2**20/3)))))
chunk = int(round(((sizes[i]**2))//(row**2)))
repfact = len(range(0,sizes[i],row))
divisions = [t for t in range(0,sizes[i],row)]
divisions2 = np.array(np.repeat(divisions, repfact), dtype=np.intc)
divisions1 = np.array((divisions * repfact), dtype=np.intc)
start = time.time()
hw14.matMult_block2(X, X, outmat, n, divisions1, divisions2, row)
print(outmat)
