import os
import numpy as np
import hw13
import hw14
import time
import csv
import math
import random
from scipy.linalg.blas import dgemm


###########################
# HW 1 QUESTION 4
###########################
nthreads = [2, 4, 8, 16, 32, 64]
sizes = [2**6, 2**10, 2**16]
iter = range(len(sizes))

fn_matmat = "matmat.csv"
with open(fn_matmat, 'w+') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Algorithm','p','2^6','2^10','thingy'])
    f.close()


start = []
dgemm_time = []
serial_time = []
gflopsPerSec = []
operations_serial = []
operations_block = []
parallel_time_naive = []
parallel_time_block = []
    
for n in nthreads:
    for i in iter:
        random.seed(5555)
        X = Y = outmat = np.zeros((sizes[i],sizes[i]))
        
        operations.append(2 * (i**3))
        
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                X[j,k] = random.gauss(0,1)
                  Y[j,k] = random.gauss(0,1)

        # Linear comparison between dgemm and cython function
        start = time.time()
        dgemm(alpha=1.,a=X,b=Y)
        dgemm_time.append(time.time()-start)
        
        start = time.time()
        hw14.matMult_serial(X, Y, outmat, n)
        serial_time.append(time.time()-start)
        
        #Naive parallel algorithm without blocking
        outmat = np.zeros((sizes[i],sizes[i]))
        row = round(23*100*1000 / 8/(sizes[i]/2))
        chunk = (2*(sizes[i]**2))/(row**2)
        if chunk < n:
            chunk = n
        start = time.time()
        hw14.matMult_thread(X, Y, outmat, n, chunk)
        parallel_time_naive.append(time.time() - start)
        
        #Naive parallel algorithm with blocking
        outmat = np.zeros((sizes[i],sizes[i]))
        row = round(23*100*1000 / 8/(sizes[i]/2))
        chunk = (2*(sizes[i]**2))/(row**2)
        if chunk < n:
            chunk = n
        start = time.time()
        hw14.matMult_block(X, Y, outmat, n, chunk)
        parallel_time_block.append(time.time() - start)
        operations_block.append(


    serial_time.insert(0,"Serial Times")
    parallel_time_naive.insert(0,"Parallel Naive Times")
    parallel_time_block.insert(0,"Parallel Block Times")
    serial_time.insert(1,n)
    parallel_time_naive.insert(1,n)
    parallel_time_block.insert(1,n)
    

    with open(fn_matmat, 'a') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow([str(i) for i in serial_time])
        writer.writerow([str(i) for i in parallel_time_naive])
        writer.writerow([str(i) for i in parallel_time_block])
        f.close()

exit()
