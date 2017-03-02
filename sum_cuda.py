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
    
    __global__ void parallel_sum_gpu(float *in_data, float *out_data) {
        extern __shared__ float data[];
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x
        unsigned int tid = threadIdx.x;
        
        data[tid] = in_data[i];
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
    
        if (tid == 0) out_data[blockIdx.x] = data[0];
    }
    """)

func = mod.get_function("parallel_sum_gpu")
#func = mod.get_function("doublify")


parallel_timings_gpu = []
start = 0
myarray = []

#for i in range(len(sizes)):
i=0
myarray = np.ones((sizes[i],), dtype=np.int_)
in_data = myarray.astype(np.float32)
out_sum = np.empty_like(in_data)


in_data_gpu = cuda.mem_alloc(in_data.nbytes)
cuda.memcpy_htod(in_data_gpu, in_data)
start = time.time()
func(in_data_gpu, block=(64,1,1), grid=(1,1,1))
cuda.memcpy_dtoh(out_sum, in_data_gpu)
parallel_timings_gpu.append(time.time()-start)
print(myarray_sum)




