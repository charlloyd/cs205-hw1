
from cython.parallel import parallel, prange

# Serial summation
def serial_summation(a):
    sums = a[0]
    i = 1
    N = len(a)
    while i < N:
        sums += a[i]
        i += 1
        
    return sums


# Parallelize summation using Cython
def parallel_sum(double [:] a):
    cdef double sum = 0;
    cdef int i = 0;
    
    for i in prange(len(a), schedule='dynamic'):
        sum += a[i];
    return sum;

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

        
        


