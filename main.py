import numpy as np
import hw13
import time

# test cases
sizes =  [2**6, 2**10, 2**20, 2**32]
#lists = [np.asarray([1]*size) for size in sizes] #each element is a list of specified size

parallel_timings = []
serial_timings = []
start = 0

for i in range(len(sizes)):
  myarray = np.asarray([1]*sizes[i])
  start = time.time()
  print(hw13.parallel_sum(myarray))
  parallel_timings.append(time.time()-start)
  start = time.time()
  print(hw13.serial_summation(myarray))
  serial_timings.append(time.time()-start)
  
with open("parallel.csv", 'w') as f:
    for s in parallel_timings:
        f.write(s + '\n')

with open("parallel.csv", 'r') as f:
    parallel_timings = [line.rstrip('\n') for line in f]
