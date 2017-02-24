import numpy as np
import hw13
import time
import csv
import openmp


# test cases
sizes =  [2**6, 2**10, 2**20]#, 2**32]

parallel_timings_naive = []
parallel_timings_thread = []
serial_timings = []
start = 0
myarray = []


for i in range(len(sizes)):
    myarray = np.ones((sizes[i],), dtype=np.int_)
    start = time.time()
    print(hw13.parallel_sum(myarray))
    parallel_timings_naive.append(time.time()-start)
    start = time.time()
    print(hw13.serial_summation(myarray))
    serial_timings.append(time.time()-start)
    start = time.time()
    print(hw13.parallel_sum_thread(myarray))
    parallel_timings_thread.append(time.time()-start)


parallel_timings_naive.insert(0,"Parallel Naive CPU Times")
parallel_timings_naive.insert(0,"Parallel Chunked CPU Times")
parallel_timings_naive.insert(0,"Serial CPU Times")
colnames = ["Algorithm"]
colnames.append(sizes)
colnames.append(sizes)
  
with open("timings_.csv", 'w', newline='') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow([str(i) for i in colnames])
    writer.writerow([str(i) for i in parallel_timings_naive])
    writer.writerow([str(i) for i in parallel_timings_thread])
    writer.writerow([str(i) for i in serial_timings])
    f.close()

exit()
