#!python
#cython: boundscheck=False
from cython.parallel import parallel, prange

# Serial summation
def serial_summation(long [:] a):
    cdef long sums
    cdef int i, N 
    
    N = a.shape[0]
    sums = a[N]
    
    for i in range(N-1):
        sums += a[i]
        
    return sums

# Parallelize summation using Cython
def parallel_sum(long [:] a):
    cdef long sums
    cdef int i, N
    
    N = a.shape[0]
    sums = a[N]
    
    for i in prange(N-1, nogil=True, schedule='dynamic'):
        sums += a[i];
    return sums;
    
#     with nogil:
#         for i in prange(N, schedule='dynamic'):
#             sums += a[i]
#     return sums

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

        
        


