#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a

from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
cimport openmp
import numpy as np

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

# Serial summation
cpdef  long  serial_summation( long[:] a):
    cdef  long  sums = a[0]
    cdef size_t i
    
    for i in range(1,a.shape[0]):
        sums += a[i]
        
    return sums

# Parallelize summation using Cython
cpdef long parallel_sum( long[:] a):
    cdef  long  sums = a[0]
    cdef size_t i
    
    for i in prange(1, a.shape[0], nogil=True, schedule='dynamic'):
        sums += a[i];
    return sums;
    

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

# Attempt at more cost effective Sum
cpdef long[:] parallel_sum_thread( long[:] data):
    nthreads = openmp.omp_get_num_threads()
    cdef double* buf = <double*>malloc(nthreads * sizeof(double))
    cdef double* threadbuf
    cdef unsigned int N = data.shape[0]
    cdef  long[::] temp_data = data
    cdef unsigned int tid, s
    cdef long sums
    cdef double* test

    sums=0

    with nogil, parallel():
        tid = threadid()
        threadbuf = buf + tid # thread setup

        for s in prange(N/2, N):
            if tid < s:
                temp_data[tid] += temp_data[tid + s];

        if tid < 32:
            test = threadbuf +temp_data[tid]
            temp_data[tid] += temp_data[tid + 32];
            temp_data[tid] += temp_data[tid + 16];
            temp_data[tid] += temp_data[tid + 8];
            temp_data[tid] += temp_data[tid + 4];
            temp_data[tid] += temp_data[tid + 2];
            temp_data[tid] += temp_data[tid + 1];
        with gil:
            print(deref(test))
    free(buf)
    return temp_data


