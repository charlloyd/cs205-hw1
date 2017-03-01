import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

sizes =  [2**6, 2**10]#, 2**20, 2**32]

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
    
    __global__ void parallel_sum_gpu(float *in_data) {
        extern __shared__ float data[];
        unsigned int tid = threadIdx.x + threadIdx.y;
        
        data[tid] = in_data[tid];
        __syncthreads();
        
        for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
            if (tid < s) {
                data[tid] += data[tid + s];
            }
            __syncthreads();
        }
        
        if (tid < 32) {
            data[tid] += data[tid + 32];
            data[tid] += data[tid + 16];
            data[tid] += data[tid + 8];
            data[tid] += data[tid + 4];
            data[tid] += data[tid + 2];
            data[tid] += data[tid + 1];
        }
    
        if (tid == 0) in_data[0] = data[0];
    }
    """)

func = mod.get_function("parallel_sum_gpu")


parallel_timings_gpu = []
start = 0
myarray = []

for i in range(len(sizes)):
myarray = np.ones((sizes[i],), dtype=np.int_)
myarray = myarray.astype(np.float32)
myarray_sum = np.empty_like(myarray)


myarray_gpu = cuda.mem_alloc(myarray.nbytes)
cuda.memcpy_htod(myarray_gpu, myarray)
#cuda.memcpy_htod(myarray_gpu, myarray_sum)
start = time.time()
func(myarray_gpu, block=(64,1,1))
cuda.memcpy_dtoh(myarray_sum, myarray_gpu)
parallel_timings_gpu.append(time.time()-start)
print(myarray_sum)




