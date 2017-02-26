#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a

from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free

# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults

###########################
# 4. matrix multiplication
###########################

cpdef double matMult_naive(double[::,::] mat, double[:] vec, int nthreads):
    cdef unsigned int N = vec.shape[0]
    cdef unsigned int J = mat.shape[1]
    cdef unsigned int s,j, n
    cdef unsigned int chunk = N/nthreads
    cdef double out[N]

    for n in range(N):
        out[n] = 0

    for s in prange(N, nogil=True, num_threads=nthreads, schedule='dynamic'):
        for j in prange(J, num_threads=nthreads, schedule='dynamic'):
            out[s] += mat[s,j] * vec[j]

    return out

cpdef double matMult_thread(double[::,::] mat, double[:] vec, int nthreads):
    cdef unsigned int N = vec.shape[0]
    cdef unsigned int J = mat.shape[1]
    cdef unsigned int s, j, n
    cdef unsigned int chunk = N/nthreads
    cdef double out[N]

    for n in range(N):
        out[n] = 0

    for s in prange(0, N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='guided'):
        for j in range(J):
            out[s] += mat[s,j] * vec[j]

    return out
