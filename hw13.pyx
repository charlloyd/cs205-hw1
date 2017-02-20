#!python
#cython: boundscheck=False
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
def parallel_sum(long [:] a):
    cdef long sums
    cdef int i, N
    
    i=0
    N = a.shape[0]
    sums = 0
    
    with nogil:
        for i in prange(N, schedule='dynamic'):
            sums += a[i]
    return sums

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

        
        


