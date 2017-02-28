#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a

from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free, rand
cimport numpy as np


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

    for n in prange(N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='static'):
        for k in range(K):
            for j in range(J):
                out[n,k] += X[n,j] * Y[j,k]
    return 0

cdef int mmb(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads, int[:,:] step, int S, int T, int chunk):
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
        for t in range(T):
            for s in range(S):
                for a in range(chunk):
                    for b in range(J):
                        A[a*J + b] = X[a + step[tid,s],b]
                        B[a*J + b] = Y[b,t]
                for k in range(chunk):
                    for j in range(J):
                        C[k] = C[k] + A[k*J + j] * B[k*J + j]
                for n in prange(nthreads):
                    for k in range(chunk):
                        out[k + step[tid,s],t] +=  C[k]
        free(A)
        free(B)
        free(C)
    return 0

def matMult_block(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads, int[:,:] step, int S, int T, int chunk)
    cdef int chunk = round(23*100*1000 / 8/(sizes[i]*2))
    step = [idx for idx in range(0,sizes[i],chunk)]
    step = np.array(step, dtype=np.intc)
    step

    mmb(
