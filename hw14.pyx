#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a

from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free, rand


# DON'T USE NEGATIVE INDEXING!!! Turning this option off makes code faster, 
# but means python style negative indexing will cause segfaults




###########################
# 4. matrix multiplication
###########################

cpdef int matMult_serial(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int J = Y.shape[0]
    cdef unsigned int K = Y.shape[1]
    cdef unsigned int k, j, n

    for n in range(N):
        for k in range(K):
            for j in range(J):
                out[n,j] += X[n,j] * Y[j,k]
    return 0

cpdef int matMult_naive(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int J = Y.shape[0]
    cdef unsigned int K = Y.shape[1]
    cdef unsigned int k, j, n

    for n in prange(N, nogil=True, num_threads=nthreads, schedule='dynamic'):
        for k in range(K):
            for j in range(J):
                out[n,j] += X[n,j] * Y[j,k]
    return 0


cpdef int matMult_thread(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int J = Y.shape[0]
    cdef unsigned int K = Y.shape[1]
    cdef unsigned int k, j, n
    cdef unsigned int chunk = N/nthreads

    for n in prange(N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='guided'):
        for k in range(K):
            for j in range(J):
                out[n,k] += X[n,j] * Y[j,k]
    return 0

cpdef int matMult_block(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int J = Y.shape[0]
    cdef unsigned int K = Y.shape[1]
    cdef unsigned int f, g, h, i j, k, l, m, n, tid
    cdef unsigned int chunk = <int>((2.3*1000*1000 / sizeof(double))/(N*2))
    cdef double *A = <double *>(malloc (N * chunk * sizeof(double)))
    cdef double *B = <double *>(malloc (N * chunk * sizeof(double)))
    cdef double *C = <double *>(malloc (N * chunk * sizeof(double)))
    cdef int[::] step = range(0,N, chunk)
    cdef int n_threads

    n_threads = min(nthreads, len(step))

    with nogil, parallel(num_threads=n_threads):
        tid = threadid()
        for f in step:
            for g in range(chunk):
                for h in range(chunk):
                    A[f + g + h] = X[step[tid] + g, h + f]
                    B[f + g + h] = Y[step[tid] + g, h + f]
            for i in range(chunk):
                for j in range(chunk):
                    C[] = A[] * B[]

            for n in prange(N):
                for m in range(chunk):
                    out[n,k+m] += C[step[tid] + m]

        free(A)
        free(B)
        free(C)
    return 0


