import os
import numpy as np
import hw14
import time
import csv
import math
import random
from scipy.linalg.blas import dgemm

###########################
# HW 1 QUESTION 4
###########################

# set number of threads
nthreads = [2, 4, 8,]# 16, 32, 64]

# over-write files
fn_matmat = "matmat.csv"
with open(fn_matmat, 'w+') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Algorithm','p','2^6','2^10','thingy'])
    f.close()

# define sizes for matrix multiplication
sizes = [2**6,]# 2**10, 2**16]
iter = range(len(sizes))

# main multiplication loop
for n in nthreads:
    
    print(n)
    
    # initialize variables
    gflopsPerSec = []
    serial_time = []
    dgemm_time = []
    parallel_time_naivedyn = []
    parallel_time_chunked = []
    parallel_time_block1 = []
    parallel_time_block2 = []
    operations_serial = []
    operations_chunked = []
    operations_block1 = []
    operations_block2 = []

    
    for i in iter:
        random.seed(5555)
        X = Y = outmat = np.zeros((sizes[i],sizes[i]))
        operations_serial.append(2 * (i**3))
        X = np.random.randn(sizes[i],sizes[i])
        Y = np.random.randn(sizes[i],sizes[i])

        # serial matrix multiplication (3 loops)
        start = time.time()
        hw14.matMult_serial(X, Y, outmat, n)
        serial_time.append(time.time()-start)
        
        # serial matrix multiplication - DGEMM
        start = time.time()
        dgemm(alpha=1.,a=X,b=Y)
        dgemm_time.append(time.time()-start)
        
        # naive dynamic parallel algorithm (no blocking)
        start = time.time()
        hw14.matMult_serial(X, Y, outmat, n)
        parallel_time_naivedyn.append(time.time()-start)
        
        # chunked parallel algorithm (no blocking)
        outmat = np.zeros((sizes[i],sizes[i]))
        row =  int(round(np.floor((np.sqrt(16*2**20/3)))))
        chunk = int(round(((sizes[i]**2))//(row**2)))
        if chunk ==0:
            chunk = 1
        while (sizes[i] % chunk) > 0 :
            chunk -= 1
        start = time.time()
        hw14.matMult_thread(X, Y, outmat, n, chunk)
        parallel_time_chunked.append(time.time() - start)
        operations_chunked.append(4 * (i**3)/chunk + 2* (i**2)/chunk)
        
        # Parallel algorithm with blocking
        outmat = np.zeros((sizes[i],sizes[i]))
        row =  int(round(np.floor((np.sqrt(16*2**20/3)))))
        chunk = int(round(((sizes[i]**2))//(row**2)))
        if chunk ==0:
            chunk = 1
        while (sizes[i] % chunk) > 0 :
            chunk -= 1
        if chunk < n:
            chunk = n
            row = int(np.ceil(np.sqrt(sizes[i]**2/n)))
        repfact = len(range(0,sizes[i],row))
        step1 = step2 = np.zeros((n,int(np.ceil(repfact**2/n))), dtype=np.intc)
        divisions = [t for t in range(0,sizes[i],row)]
        divisions2 = np.repeat(divisions, repfact)
        divisions1 = (divisions * repfact)
        count =  0
        for jdx in range(step1.shape[1]):
            for idx in range(step1.shape[0]):
                if count < divisions2.shape[0]:
                    step1[idx,jdx] = divisions1[count]
                    step2[idx,jdx] = divisions2[count]
                    count += 1
                else:
                    step1[idx,jdx] = 2**25
                    step2[idx,jdx] = 2**25
        start = time.time()
        #hw14.matMult_block(X, Y, outmat, n, step1, step2, row)
        parallel_time_block1.append(time.time() - start)
        operations_block1.append(4 * (i**3)/chunk + 2* (i**2)/chunk )
        
        # Parallel algorithm with all cores working on same block
        outmat = np.zeros((sizes[i],sizes[i]))
        row =  int(round(np.floor((np.sqrt(16*2**20/3)))))
        chunk = int(round(((sizes[i]**2))//(row**2)))
        if chunk < n:
            chunk = n
            row = int(np.ceil(np.sqrt(sizes[i]**2/n)))  
        repfact = len(range(0,sizes[i],row))
        divisions = [t for t in range(0,sizes[i],row)]
        divisions2 = np.repeat(divisions, repfact)
        divisions1 = (divisions * repfact)
        start = time.time()
        #hw14.matMult_block2(X, Y, outmat, n, divisions1, divisions2, row)
        parallel_time_block2.append(time.time() - start)
        operations_block2.append(4 * (i**3)/chunk + 2* (i**2)/chunk )
        
    # prep before writing
    serial_time.insert(0,"Serial Times")
    dgemm_time.insert(0,"DGEMM Times")
    parallel_time_naivedyn.insert(0,"Parallel Naive Dynamic Times")
    parallel_time_chunked.insert(0,"Parallel Naive Chunked Times")
    parallel_time_block1.insert(0,"Parallel Block1 Times")
    parallel_time_block2.insert(0,"Parallel Block1 Times")
    operations_serial.insert(0, "Serial Operations")
    serial_time.insert(1,n)
    dgemm_time.insert(1,n)
    parallel_time_naivedyn.insert(1,n)
    parallel_time_chunked.insert(1,n)
    parallel_time_block1.insert(1,n)
    parallel_time_block2.insert(1,n)
    operations_chunked.insert(0, "Parallel Naive Chunked Operations")
    operations_block1.insert(0, "Parallel Block1 Operations")
    operations_block2.insert(0, "Parallel Block2 Operations")
    
    # write results to csv
    with open(fn_matmat, 'a') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow([str(i) for i in serial_time])
        writer.writerow([str(i) for i in dgemm_time])
        writer.writerow([str(i) for i in parallel_time_naivedyn])
        writer.writerow([str(i) for i in parallel_time_chunked])
        writer.writerow([str(i) for i in parallel_time_block1])
        writer.writerow([str(i) for i in parallel_time_block2])
        writer.writerow([str(i) for i in operations_serial])
        writer.writerow([str(i) for i in operations_chunked])
        writer.writerow([str(i) for i in operations_block1])
        writer.writerow([str(i) for i in operations_block2])
        f.close()

exit()
