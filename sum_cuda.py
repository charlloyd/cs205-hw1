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
        unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;
        unsigned int i = threadIdx.x;

        data[i] = in_data[tid];
        
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
    
    __global__ void getGlobalIdx_1D_1D(int *in_data) {
        int idx = threadIdx.x ;
        in_data[idx] = idx;
    }
    """)

func = mod.get_function("parallel_sum_gpu")
func2 = mod.get_function("getGlobalIdx_1D_1D")
#func = mod.get_function("doublify")


parallel_timings_gpu = []
start = 0
myarray = []

#for i in range(len(sizes)):
i=0
myarray = np.zeros((sizes[i],), dtype=np.int_) -70
in_data = myarray.astype(np.intc)
out_data = np.empty_like(in_data, dtype=np.intc)
#N = np.array(sizes[i], dtype=np.intc)


in_data_gpu = cuda.mem_alloc(in_data.nbytes)
#out_data_gpu = cuda.mem_alloc(out_data.nbytes)
#N_gpu = cuda.mem_alloc(N.nbytes)
cuda.memcpy_htod(in_data_gpu, in_data)
#cuda.memcpy_htod(out_data_gpu, out_data)
#cuda.memcpy_htod(N_gpu, N)

start = time.time()
func2(in_data_gpu, block=(20,1,1), grid=(1,1,1))
#func(in_data_gpu, out_data_gpu, block=(1,1,1), grid=(64,1,1))
cuda.memcpy_dtoh(out_data, in_data_gpu)
parallel_timings_gpu.append(time.time()-start)
print(out_data)


i=0
myarray = np.ones((sizes[i],), dtype=np.int_)
in_data = myarray.astype(np.float32)
out_data = np.zeros((sizes[i],), dtype=np.float32)
#N = np.array(sizes[i], dtype=np.intc)


in_data_gpu = cuda.mem_alloc(in_data.nbytes)
out_data_gpu = cuda.mem_alloc(out_data.nbytes)
#N_gpu = cuda.mem_alloc(N.nbytes)
cuda.memcpy_htod(in_data_gpu, in_data)
cuda.memcpy_htod(out_data_gpu, out_data)
#cuda.memcpy_htod(N_gpu, N)

start = time.time()
func(in_data_gpu, out_data_gpu, block=(sizes[i],1,1), grid=(1,1,1), shared=in_data.nbytes*sizes[i])
cuda.memcpy_dtoh(out_data, out_data_gpu)
parallel_timings_gpu.append(time.time()-start)
print(out_data)




