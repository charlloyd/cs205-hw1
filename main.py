import numpy
import hw13

# test cases
sizes = [2**6] #, 2**10, 2**20, 2**32]
lists = [[1]*size for size in sizes] #each element is a list of specified size

print(hw13.summation(lists))
