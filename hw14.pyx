#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# distutils: extra_link_args = -fopenmp


from cython.parallel cimport parallel, prange, threadid
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free, rand
cimport numpy as np
import numpy as np


###########################
# 4. matrix-matrix multiplication
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
    cdef size_t k, j, n

    for n in prange(N, nogil=True, num_threads=nthreads, schedule='dynamic'):
        for k in range(K):
            for j in range(J):
                out[n,j] += X[n,j] * Y[j,k]
    return 0

cpdef int matMult_thread(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads, int chunk):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int J = Y.shape[0]
    cdef unsigned int K = Y.shape[1]
    cdef size_t k, j, n

    for n in prange(N, nogil=True, num_threads=nthreads, chunksize=chunk, schedule='static'):
        for k in range(K):
            for j in range(J):
                out[n,k] += X[n,j] * Y[j,k]
    return 0

cdef void reduce(double[::,::] out, double * C, int s, int t, int N, int stop) nogil:
    cdef size_t k,j

    for k in range(N):
        for j in range(N)
            if (s+k < stop) & (t+j < stop):
                out[s+k,t+j] += C[N*k + j]

cdef void mmb(double[::,::] X, double[::,::] Y, double[::,::] out, int nthreads, int[::,::] step, int S, int chunk, int N, int J, int K):
    cdef size_t a, b, k, j, n, s,t
    cdef int tid
    cdef double *A
    cdef double *B
    cdef double *C

    with nogil, parallel(num_threads = nthreads):
        tid = threadid()
        A = <double *>(malloc (10*J * chunk * sizeof(double)))
        B = <double *>(malloc (10*J * chunk * sizeof(double)))
        C = <double *>(malloc (10*chunk * chunk * sizeof(double)))
        for s in range(S):
            for a in range(chunk):
                if ((a + step[tid,s]) < N) & ((a + step2[tid,s])<K):
                    for b in range(J):
                        A[a*J + b] = X[a + step1[tid,s], b]
                        B[a*J + b] = Y[b, a + step2[tid,s]]
            for k in range(chunk):
                for j in range(chunk):
                    if ((k + step[tid,s]) < N) & ((j + step2[tid,s])<K):
                        for t in range(J):
                            C[k*J + j] = C[k*J + j] + A[k*J + t] * B[j*J + t]
            for n in prange(nthreads):
                reduce(out, C, step1[tid,s], step2[tid,s], chunk, N)
        free(A)
        free(B)
        free(C)

def matMult_block(double[::,::] X, double[::,::] Y, out int nthreads, int[::, ::] step1, int[::, ::] step2, int chunk):
    cdef int S = step1.shape[1]
    cdef int K = Y.shape[1]
    cdef int N = X.shape[0]
    cdef int J = Y.shape[0]
    cdef double[::,::] Xc = X
    cdef double[::,::] Yc = Y
    cdef double[::,::] outC = out
    cdef int nt = nthreads
    cdef int[::,::] stepC1 = step1
    cdef int[::,::] stepC2 = step2
    cdef int chunkC = chunk

    mmb(Xc, Yc, outC, nt, stepC, S, chunkC, N, J, K)
    return outC



