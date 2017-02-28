#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# distutils: language = c++



###########################
## HW 1 QUESTION 3
###########################

from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libcpp.stl cimport malloc
cimport numpy as np
import numpy as np
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

###########################
# SUMMATION
###########################

# Serial summation
cpdef long serial_summation(long[:] a):
    cdef  long  sums = a[0]
    cdef size_t i
    
    for i in range(1,a.shape[0]):
        sums += a[i]
        
    return sums

# Parallelize summation using Cython
cpdef long parallel_sum(long[:] a, int nthreads):
    cdef  long  sums = a[0]
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

    for s in prange(N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='static'):
        sums += data[s]
        
    return sums


###########################
# MATRIX VECTOR MULTIPLICATION
###########################
cpdef int vecmatMult_serial(double[::,::] mat, double[::] vec, double[::] out):
    cdef unsigned int N = vec.shape[0]
    cdef unsigned int J = mat.shape[1]
    cdef unsigned int j, n

    for n in range(N):
        for j in range(J):
            out[n] += mat[n,j] * vec[j]
    return 0


cpdef int vecmatMult_naive(double[::,::] mat, double[::] vec, double[::] out, int nthreads):
    cdef unsigned int N = vec.shape[0]
    cdef unsigned int J = mat.shape[1]
    cdef int n,j

    for n in prange(N, nogil=True, num_threads=nthreads, schedule='dynamic'):
        for j in prange(J, num_threads=nthreads, schedule='dynamic'):
            out[n] += mat[n,j] * vec[j]
    return 0

cpdef int vecmatMult_thread(double[::,::] mat, double[::] vec, double[::] out, int nthreads, int chunk):
    cdef unsigned int N = vec.shape[0]
    cdef unsigned int J = mat.shape[1]
    cdef size_t j
    cdef int n

    for n in prange(0, N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='static'):
        for j in range(J):
            out[n] += mat[n,j] * vec[j]
    return 0
