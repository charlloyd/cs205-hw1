#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False

from cython.parallel import parallel, prange

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

# Serial summation
cdef serial_summation(long [:] a):
    cdef long sums
    cdef size_t i
    
    sums = a[0]
    
    for i in range(1,a.shape[0]):
        sums += a[i]
        
    return sums

# Parallelize summation using Cython
cdef parallel_sum(long [:] a) nogil:
    cdef long sums
    cdef size_t i
    
    sums = a[0]
    
    for i in prange(1, a.shape[0], nogil=True, num_threads=64, schedule='static'):
        sums += a[i];
    return sums;
    

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

        
        


