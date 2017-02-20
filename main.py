import numpy as np
import hw13

# test cases
sizes =  [2**6, 2**10, 2**20, 2**32]
lists = [np.asarray([1]*size) for size in sizes] #each element is a list of specified size

for i in range(len(lists)):
  print(hw13.parallel_sum(lists[i]))

for i in range(len(lists)):
  print(hw13.serial_summation(lists[i]))
