#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a

###########################
## HW 1 QUESTION 3
###########################

from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

###########################
# SUMMATION
###########################

# Serial summation
cpdef long serial_summation(long[:] a):
    cdef  long  sums = a[0]
    
    for i in xrange(1, a.shape[0]):
        sums += a[i]

    return sums

cpdef double serial_summation(double[:] a):
    cdef  double  sums = a[0]

    for i in xrange(1, a.shape[0]):
        sums += a[i]

    return sums

# Parallelize summation using Cython
cpdef long parallel_sum(long[:] a, int nthreads):
    cdef  long  sums = a[0]
    cdef int i
    
    for i in prange(1, a.shape[0], nogil=True, schedule='dynamic', num_threads=nthreads):
        sums += a[i];
        
    return sums

cpdef double parallel_sum(double[:] a, int nthreads):
    cdef  double  sums = a[0]
    cdef int i

    for i in prange(1, a.shape[0], nogil=True, schedule='dynamic', num_threads=nthreads):
        sums += a[i];

    return sums

# Optimize this parallelization
# adjust the number of threads to make the algorithm cost optimal

# Attempt at more cost effective Sum
cpdef long parallel_sum_thread(long[::] data, int nthreads):
    cdef unsigned int N = data.shape[0]
    cdef int s
    cdef unsigned int chunk = N/nthreads
    cdef long sums = 0

    for s in prange(N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='guided'):
        sums += data[s]
        
    return sums

cpdef double parallel_sum_thread(long[::] data, int nthreads):
    cdef unsigned int N = data.shape[0]
    cdef int s
    cdef unsigned int chunk = N/nthreads
    cdef double sums = 0

    for s in prange(N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='guided'):
        sums += data[s]

    return sums

###########################
# MATRIX VECTOR MULTIPLICATION
###########################

# note -- I moved 2 functions for matrix vector multiplication into hw14.pyx so i could get this file to compile
