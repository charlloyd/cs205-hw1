import numpy as np
import hw13
import time
import csv

# test cases
sizes =  [2**6, 2**10, 2**20]
#lists = [np.asarray([1]*size) for size in sizes] #each element is a list of specified size

parallel_timings = []
serial_timings = []
start = 0

for i in range(len(sizes)):
  myarray = np.ascontiguousarray([1]*sizes[i], dtype=np.int_)
  start = time.time()
  print(hw13.parallel_sum(myarray))
  parallel_timings.append(time.time()-start)
  start = time.time()
  print(hw13.serial_summation(myarray))
  serial_timings.append(time.time()-start)
  
with open("timings.csv", 'wb') as f:
    writer = csv.writer(f)
    writer.writerows([parallel_timings, serial_timings])
    f.close()
