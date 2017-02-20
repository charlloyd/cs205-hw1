#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

from cython.parallel import parallel, prange

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

# Serial summation
def serial_summation(long [:] a):
    cdef long sums
    cdef int N
    cdef size_t i
    
    N = a.shape[0]
    sums = a[N-1]
    
    for i in range(N-1):
        sums += a[i]
        
    return sums

# Parallelize summation using Cython
def parallel_sum(long [:] a):
    cdef long sums
    cdef int N
    cdef size_t i
    
    N = a.shape[0]
    sums = a[N-1]
    
    for i in prange(N-1, nogil=True, num_threads=16, schedule='static'):
        sums += a[i];
    return sums;
    

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

        
        


