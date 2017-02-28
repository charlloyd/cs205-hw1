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
# HW 1 QUESTION 3
###########################

# over-write files
fn_sum = "summation.csv"
fn_matvec = "matvec.csv"
fn_matmat = "matmat.csv"
with open(fn_sum, 'w+') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Algorithm','p','2^6','2^10','2^20','thingy'])
    f.close()
with open(fn_matvec, 'w+') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Algorithm','p','2^6','2^10','thingy'])
    f.close()
with open(fn_matmat, 'w+') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Algorithm','p','2^6','2^10','thingy'])
    f.close() # not written downbelow


# set number of threads
nthreads = [2, 4, 8, 16, 32]

# main loop for different numbers of threads
for n in nthreads:
    print(n)
    ### SUMMATION ###
    
    # initialize variables
    serial_timings = []
    parallel_timings_naive = []
    parallel_timings_thread = []
    serial_result=[]
    parallel_result_naive =[]
    parallel_result_thread=[]
    myarray = []
    
    # define sizes for summation
    sizes =  [2**6, 2**10, 2**20]# 2**32]
    iter = range(len(sizes))
    
    # main summation loop
    for i in iter:
        myarray = np.ones((sizes[i],), dtype=np.int_)
        
        # serial summation algorithm
        start = time.time()
        serial_result.append(hw13.serial_summation(myarray))
        serial_timings.append(time.time()-start)
        
        # parallel naive summation algorithm
        start = time.time()
        parallel_result_naive.append(hw13.parallel_sum(myarray, n))
        parallel_timings_naive.append(time.time()-start)
        
        # parallel thread ("guided") summation algorithm
        start = time.time()
        parallel_result_thread.append(hw13.parallel_sum_thread(myarray, n))
        parallel_timings_thread.append(time.time()-start)

    # timings
    serial_timings.append(np.array_equal(sizes, serial_result))
    parallel_timings_naive.append(np.array_equal(sizes, parallel_result_naive))
    parallel_timings_thread.append(np.array_equal(sizes, parallel_result_thread))

    # speedup
    parallel_spd_naive = [serial_timings[i]/parallel_timings_naive[i] for i in iter]
    parallel_spd_naive.append(parallel_timings_naive[-1])
    parallel_spd_thread = [serial_timings[i]/parallel_timings_thread[i] for i in iter]
    parallel_spd_thread.append(parallel_timings_thread[-1])

    # efficiency
    parallel_eff_naive = [parallel_spd_naive[i]/n for i in iter]
    parallel_eff_naive.append(parallel_timings_naive[-1])
    parallel_eff_thread = [parallel_spd_thread[i]/n for i in iter]
    parallel_eff_thread.append(parallel_timings_thread[-1])

    # prep before writing
    serial_timings.insert(0,"Serial Times")
    parallel_timings_naive.insert(0,"Parallel Naive Times")
    parallel_timings_thread.insert(0,"Parallel Guided Times")
    parallel_spd_naive.insert(0,"Parallel Naive Speed-up")
    parallel_spd_thread.insert(0,"Parallel Guided Speed-up")
    parallel_eff_naive.insert(0,"Parallel Naive Efficiency")
    parallel_eff_thread.insert(0,"Parallel Guided Efficiency")
    serial_timings.insert(1,n)
    parallel_timings_naive.insert(1,n)
    parallel_timings_thread.insert(1,n)
    parallel_spd_naive.insert(1,n)
    parallel_spd_thread.insert(1,n)
    parallel_eff_naive.insert(1,n)
    parallel_eff_thread.insert(1,n)
    
    # write results to csv
    
    with open(fn_sum, 'a') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow([str(i) for i in serial_timings])
        writer.writerow([str(i) for i in parallel_timings_naive])
        writer.writerow([str(i) for i in parallel_timings_thread])
        writer.writerow([str(i) for i in parallel_spd_naive])
        writer.writerow([str(i) for i in parallel_eff_naive])
        writer.writerow([str(i) for i in parallel_spd_thread])
        writer.writerow([str(i) for i in parallel_eff_thread])
        f.close()
    
    ### MATRIX VECTOR MULTIPLICATION ###
    
    # re-initialize arrays
    serial_timings = []
    parallel_timings_naive = []
    parallel_timings_thread = []
    serial_result=[]
    parallel_result_naive =[]
    parallel_result_thread=[]
    compare = []
    
    # re-define sizes
    sizes = [2**6, 2**10, 2**16]
    iter = range(len(sizes))
    
    # main matrix-vector multiplication loop
    for i in iter:
        random.seed(5555)
        myvec = np.zeros((sizes[i],))
        outvec = np.zeros_like(myvec)
        mymat = np.zeros((sizes[i], sizes[i]))
        
        # create a matrix and vector
        for j in range(sizes[i]):
            myvec[j] = random.gauss(0,1)
            for k in range(sizes[i]):
                mymat[j,k] = random.gauss(0,1)
        
        # NP test
        compare.append(np.dot(mymat, myvec))
        
        # serial matrix-vector multiplication algorithm
        start = time.time()
        hw13.vecmatMult_serial(mymat, myvec, outvec)
        serial_timings.append(time.time()-start)
        serial_result.append(outvec)
        
        # parallel naive matrix-vector multiplication algorithm
        outvec = np.zeros_like(myvec)
        start = time.time()
        hw13.vecmatMult_naive(mymat, myvec, outvec, n)
        parallel_timings_naive.append(time.time()-start)
        parallel_result_naive.append(outvec)
        
        # parallel thread ("guided") matrix-vector multiplication algorithm
        outvec = np.zeros_like(myvec)
        row = round(23*100*1000 / 8/(sizes[i]/2))
        chunk = (2*(sizes[i]**2))/(row**2)
        if chunk < n:
            chunk = n
        #step = [idx for idx in range(0,sizes[i],row)]
        #step = np.array(step, dtype=np.intc)
        start = time.time()
        hw13.vecmatMult_thread(mymat, myvec, outvec, n, chunk)
        parallel_timings_thread.append(time.time()-start)
        parallel_result_thread.append(outvec)
    # speedup
    parallel_spd_naive = [serial_timings[i]/parallel_timings_naive[i] for i in iter]
    parallel_spd_naive.append(parallel_timings_naive[-1])
    parallel_spd_thread = [serial_timings[i]/parallel_timings_thread[i] for i in iter]
    parallel_spd_thread.append(parallel_timings_thread[-1])
    
    # efficiency
    parallel_eff_naive = [parallel_spd_naive[i]/n for i in iter]
    parallel_eff_naive.append(parallel_timings_naive[-1])
    parallel_eff_thread = [parallel_spd_thread[i]/n for i in iter]
    parallel_eff_thread.append(parallel_timings_thread[-1])
    
    # prep before writing
    serial_timings.insert(0,"Serial Times")
    parallel_timings_naive.insert(0,"Parallel Naive Times")
    parallel_timings_thread.insert(0,"Parallel Guided Times")
    parallel_spd_naive.insert(0,"Parallel Naive Speed-up")
    parallel_spd_thread.insert(0,"Parallel Guided Speed-up")
    parallel_eff_naive.insert(0,"Parallel Naive Efficiency")
    parallel_eff_thread.insert(0,"Parallel Guided Efficiency")
    serial_timings.insert(1,n)
    parallel_timings_naive.insert(1,n)
    parallel_timings_thread.insert(1,n)
    parallel_spd_naive.insert(1,n)
    parallel_spd_thread.insert(1,n)
    parallel_eff_naive.insert(1,n)
    parallel_eff_thread.insert(1,n)
    
    # write results to csv
    with open(fn_matvec, 'a') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow([str(i) for i in serial_timings])
        writer.writerow([str(i) for i in parallel_timings_naive])
        writer.writerow([str(i) for i in parallel_timings_thread])
        writer.writerow([str(i) for i in parallel_spd_naive])
        writer.writerow([str(i) for i in parallel_eff_naive])
        writer.writerow([str(i) for i in parallel_spd_thread])
        writer.writerow([str(i) for i in parallel_eff_thread])
        f.close()


    ###########################
    # HW 1 QUESTION 4
    ###########################
    
    start = []
    dgemm_time = []
    serial_time = []
    gflopsPerSec = []
    operations = []
    parallel_time_naive = []
    parallel_time_block = []

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
        parallel_time_naive.append(time.time() - start)
        
###########################
# HW 1 QUESTION 4
###########################

start = []
dgemm_time = []
serial_time = []
gflopsPerSec = []
operations = []

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
    #matMult_serial(X, Y, outmat, 32)
    serial_time.append(time.time()-start)
    
    #Naive parallel algorithm without blocking

exit()
