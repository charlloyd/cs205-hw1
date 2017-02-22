#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a

from cython.parallel import parallel, prange, threadsavailable, threadid
from

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

# Serial summation
cpdef long serial_summation(long[:] a):
    cdef long sums = a[0]
    cdef size_t i
    
    for i in range(1,a.shape[0]):
        sums += a[i]
        
    return sums

# Parallelize summation using Cython
cpdef long parallel_sum(long[:] a):
    cdef long sums = a[0]
    cdef size_t i
    
    for i in prange(1, a.shape[0], nogil=True, schedule='dynamic'):
        sums += a[i];
    return sums;
    

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

# Attempt at more cost effective Sum
cpdef long parallel_sum_thread(long[:] data):
    cdef double* buf = <double*>malloc(threadsavailable(schedule='dynamic') * sizeof(double))
    cdef double* threadbuf
    cdef long sums = 0
    cdef int N = data.shape[0]
    cdef size_t i
    cdef threadlocal(double) temp_sum = 0

    with nogil, parallel:
        tid = threadid()
#       threadbuf = buf + threadid() # thread setup

        for s in prange(N/2, schedule='dynamic'):
            if (tid < s) {
                data[tid] += data[tid + s];
            }
            _syncthreads();

        if (tid < 32)   {
            data[tid] += data[tid + 32];
            data[tid] += data[tid + 16];
            data[tid] += data[tid + 8];
            data[tid] += data[tid + 4];
            data[tid] += data[tid + 2];
            data[tid] += data[tid + 1];
        }

    return sums;



