import numpy
import hw13

# test cases
# [2**6, 2**10, 2**20]#,
sizes =  [2**6, 2**10, 2**20, 2**32]
lists = [[1]*size for size in sizes] #each element is a list of specified size

for i in range(len(lists)):
  print(hw13.serial_summation(lists[i]))
