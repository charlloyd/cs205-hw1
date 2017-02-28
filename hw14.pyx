#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a

from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free, rand


###########################
# 3. matrix-vector multiplication
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

cpdef int matMult_block(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads, int[:,:] step1, int S, int[:,:] step2, int T, int chunk):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int J = Y.shape[0]
    cdef unsigned int K = Y.shape[1]
    cdef unsigned int a, b, k, j, n, s,t
    cdef unsigned int chunk = N/nthreads
    cdef int *tid
    cdef double *A
    cdef double *B
    cdef double *C

    with nogil, parallel(num_threads = nthreads):
        tid = threadid()
        A = <double *>(malloc (J * chunk * sizeof(double)))
        B = <double *>(malloc (J * chunk * sizeof(double)))
        C = <double *>(malloc (chunk * sizeof(double)))
        for t in T:
            for s in S:
                for a in range(chunk):
                    C[a] = 0
                    for b in range(J):
                        A[a*J + b] = X[a + step1[tid,s],b]
                        B[a*J + b] = Y[b,t]
                for k in range(chunk):
                    for j in range(J):
                        C[k] = C[k] + A[k*J + j] * B[k*J + j]
                for n in prange(nthreads):
                    for k in range(chunk):
                        out[k + step1[tid,s],t] += out[k + step1[tid,s],t] + C[k]
    return 0

