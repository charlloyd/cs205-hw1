#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

from cython.parallel import parallel, prange

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

# Serial summation
cpdef long serial_summation(long[:] a, int N):
    cdef long sums = a[0]
    cdef size_t i
    
    for i in range(1,N):
        sums += a[i]
        
    return sums

# Parallelize summation using Cython
cpdef long parallel_sum(long[:] a, int N) nogil:
    cdef long sums = a[0]
    cdef size_t i
    
    for i in prange(1, N, nogil=True, num_threads=64, schedule='static'):
        sums += a[i];
    return sums;
    

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

        
        


