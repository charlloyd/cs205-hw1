import os
import numpy as np
import hw13
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
with open(fn_sum, 'w+') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Algorithm','p','2^6','2^10','2^20','thingy'])
    f.close()
with open(fn_matvec, 'w+') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Algorithm','p','2^6','2^10','thingy'])
    f.close()

# set number of threads
nthreads = [2, 4, 8, 16, 32]#, 64]
sizes =  [2**6, 2**10, 2**20]# 2**32]
iter = range(len(sizes))
arraylist = [np.ones((sizes[i],), dtype=np.int_) for i in iter]

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
    
    
    # main summation loop
    for i in iter:
        myarray = arraylist[i]
        
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
        print(parallel_timings_thread)
        print(serial_timings)
        print(parallel_timings_naive)
    

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
# re-define sizes
sizes = [2**6, 2**10, 2**16]
iter = range(len(sizes))

veclist = [np.ones((sizes[i],), dtype=np.float64) for i in iter]
mymatlist = [np.ones((sizes[i], sizes[i]), dtype=np.float64) for i in iter]

for n in nthreads:
    print(n)
    serial_timings = []
    parallel_timings_naive = []
    parallel_timings_thread = []
    serial_result=[]
    parallel_result_naive =[]
    parallel_result_thread=[]
    compare = []


    
    
    # main matrix-vector multiplication loop
    for i in iter:
        myvec = veclist[i]
        outvec = np.zeros_like(myvec)
        mymat = mymatlist[i]
        
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
        chunk = round((2*(sizes[i]**2))/(row**2))
        if chunk < n:
            chunk = n
        #step = [idx for idx in range(0,sizes[i],row)]
        #step = np.array(step, dtype=np.intc)
        start = time.time()
        hw13.vecmatMult_thread(mymat, myvec, outvec, n, chunk)
        parallel_timings_thread.append(time.time()-start)
        parallel_result_thread.append(outvec)
        print(parallel_timings_thread)
        print(serial_timings)
        print(parallel_timings_naive)
        
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

exit()
