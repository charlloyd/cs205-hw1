import os
import numpy as np
import hw13
#import hw14
import time
import csv
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import random

###########################
# HW 1 QUESTION 3
###########################

# test cases
sizes =  [2**6, 2**10, 2**20]# 2**32]

parallel_timings_naive = []
parallel_timings_thread = []
serial_timings = []

parallel_result_naive =[]
parallel_result_thread=[]
serial_result=[]

start = 0
myarray = []
nthreads = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
#nthreads = 64
nthreads = 8

iter = range(len(sizes))

for i in iter:
    myarray = np.ones((sizes[i],), dtype=np.int_)
    
    # serial
    start = time.time()
    serial_result.append(hw13.serial_summation(myarray))
    serial_timings.append(time.time()-start)
    
    # parallel naive
    start = time.time()
    parallel_result_naive.append(hw13.parallel_sum(myarray, nthreads))
    parallel_timings_naive.append(time.time()-start)
    
    # parallel thread
    start = time.time()
    parallel_result_thread.append(hw13.parallel_sum_thread(myarray, nthreads))
    parallel_timings_thread.append(time.time()-start)

parallel_spd_naive = [serial_timings[i]/parallel_timings_naive[i] for i in iter]
parallel_spd_thread = [serial_timings[i]/parallel_timings_thread[i] for i in iter ]
parallel_eff_naive = [parallel_spd_naive[i]/nthreads for i in iter ]
parallel_eff_thread = [parallel_spd_thread[i]/nthreads for i in iter ]

parallel_timings_naive.insert(0,"Parallel Naive Times")
parallel_timings_thread.insert(0,"Parallel Chunked Times")
serial_timings.insert(0,"Serial Times")
parallel_spd_naive.insert(0,"Parallel Naive Speed-up")
parallel_spd_thread.insert(0,"Parallel Chunked Speed-up")
parallel_eff_naive.insert(0,"Parallel Naive Efficiency")
parallel_eff_thread.insert(0,"Parallel Chunked Efficiency")

colnames = ["Algorithm"]
colnames.append(sizes)
colnames.append("Pass")
parallel_timings_naive.append(np.array_equal(sizes, parallel_result_naive))
parallel_timings_thread.append(np.array_equal(sizes, parallel_result_thread))
serial_timings.append(np.array_equal(sizes, serial_result))

parallel_spd_naive.append(parallel_timings_naive[-1])
parallel_spd_thread.append(parallel_timings_thread[-1])
parallel_eff_naive.append(parallel_timings_naive[-1])
parallel_eff_thread.append(parallel_timings_thread[-1])

filename_time = "sum_timings_nthread_" + str(nthreads) + ".csv"
filename_spd = "sum_spd_nthread_" + str(nthreads) + ".csv"
filename_eff = "sum_eff_nthread_" + str(nthreads) + ".csv"
filename_results = "sum_results" + str(nthreads) + ".csv"

with open(filename_time, 'w', newline='') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow([str(i) for i in colnames])
    writer.writerow([str(i) for i in parallel_timings_naive])
    writer.writerow([str(i) for i in parallel_spd_naive])
    writer.writerow([str(i) for i in parallel_eff_naive])
    writer.writerow([str(i) for i in parallel_timings_thread])
    writer.writerow([str(i) for i in parallel_spd_thread])
    writer.writerow([str(i) for i in parallel_eff_thread])
    writer.writerow([str(i) for i in serial_timings])
    f.close()



### Matrix Vector Multiplication ###
sizes = [2**6, 2**10, 2**16]

parallel_timings_naive = []
parallel_timings_thread = []
serial_timings = []

parallel_result_naive =[]
parallel_result_thread=[]
serial_result=[]
compare = []


iter = range(len(sizes))

for i in iter:
    random.seed(5555)
    myvec = np.zeros((sizes[i],))
    outvec = np.zeros_like(myvec)
    mymat = np.zeros((sizes[i], sizes[i]))
    #
    for j in range(sizes[i]):
        myvec[j] = random.gauss(0,1)
        for k in range(sizes[i]):
            mymat[j,k] = random.gauss(0,1)
    #
    # NP test
    compare.append(np.dot(mymat, myvec))
    #
    # serial
    start = time.time()
    hw13.vecmatMult_serial(mymat, myvec, outvec)
    serial_timings.append(time.time()-start)
    serial_result.append(outvec)
    #
    # parallel naive
    outvec = np.zeros_like(myvec)
    start = time.time()
    hw13.vecmatMult_naive(mymat, myvec, outvec, nthreads)
    parallel_timings_naive.append(time.time()-start)
    parallel_result_naive.append(outvec)
    #
    # parallel thread
    outvec = np.zeros_like(myvec)
    start = time.time()
    hw13.vecmatMult_thread(mymat, myvec, outvec, nthreads)
    parallel_timings_thread.append(time.time()-start)
    parallel_result_thread.append(outvec)


parallel_eff_naive = [serial_timings[i]/parallel_timings_naive[i]/nthreads for i in iter]
parallel_eff_thread = [serial_timings[i]/parallel_timings_thread[i]/nthreads for i in iter]

parallel_timings_naive.insert(0,"Parallel Naive Times")
parallel_timings_thread.insert(0,"Parallel Chunked Times")
serial_timings.insert(0,"Serial Times")
parallel_eff_naive.insert(0,"Parallel Naive Efficiency")
parallel_eff_thread.insert(0,"Parallel Chunked Efficiency")

colnames = ["Algorithm"]
colnames.append(sizes)

filename_time = "matvec_timings_nthread_" + str(nthreads) + ".csv"

with open(filename_time, 'w', newline='') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow([str(i) for i in colnames])
    writer.writerow([str(i) for i in parallel_timings_naive])
    writer.writerow([str(i) for i in parallel_eff_naive])
    writer.writerow([str(i) for i in parallel_timings_thread])
    writer.writerow([str(i) for i in parallel_eff_thread])
    writer.writerow([str(i) for i in serial_timings])
    f.close()

###########################
# HW 1 QUESTION 4
###########################

exit()
