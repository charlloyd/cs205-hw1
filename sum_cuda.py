import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np



mod = SourceModule("""
    __global__ void doublify(float *a)
    {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
    }
    """)
