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
from libc.stdlib cimport malloc, free
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

cdef long psb(long[::] data, int nthreads, int[::] step, int chunk, int N):
    cdef size_t j, n
    cdef int s
    cdef long *sdata
    cdef long temp_sum
    cdef unsigned int tid
    cdef long sums = 0

    with nogil, parallel(num_threads=nthreads):
        tid = threadid()
        sdata = <long*>(malloc(32*chunk * sizeof(long)*nthreads))
        temp_sum = <long>(malloc(sizeof(long)*32*nthreads))

        for j in range(chunk):
            sdata[j] = data[step[tid] + j]
        for n in range(chunk):
            temp_sum = temp_sum + sdata[n]
        for s in prange(nthreads):
            sums += temp_sum
        free(sdata)
    return sums

# Attempt at more cost effective Sum
def parallel_sum_block(long[::] data, int nthreads, int[::] step, int chunk):
    cdef unsigned int N = data.shape[0]
    cdef long sums
    cdef long[::] d = data
    cdef int nt = nthreads
    cdef int[::] stepC = step
    cdef int chunkC = chunk

    sums = psb(d, nt, stepC, chunkC, N)
    return np.asarray(sums)

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

cpdef int vecmatMult_explicit(double[::,::] mat, double[::] vec, double[::] out, int nthreads, int[:] step, int chunk):
    cdef unsigned int N = vec.shape[0]
    cdef unsigned int J = mat.shape[1]
    cdef size_t j, k, f, g, v, s
    cdef int t
    cdef unsigned int tid
    cdef double *vecChunk
    cdef double *matChunk
    cdef double *temp

    with nogil, parallel(num_threads=nthreads):
        tid = threadid()
        vecChunk = <double *>(malloc (N*32 * sizeof(double)))
        matChunk = <double *>(malloc (N*32 * chunk * sizeof(double)))
        temp = <double *>(malloc (chunk * sizeof(double)))

        for f in range(chunk):
            for g in range(J):
                matChunk[f*J + g] = mat[step[tid] + f, g]
        for v in range(N):
            vecChunk[v] = vec[v]
        for k in range(chunk):
            for j in range(N):
                temp[k] = temp[k] + matChunk[k*J + j] * vecChunk[j]
        for t in prange(nthreads):
            for s in range(chunk):
                out[step[tid] + s] += temp[s]

        free(matChunk)
        free(temp)
        free(vecChunk)
    return 0

