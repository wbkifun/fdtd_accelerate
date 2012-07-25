#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

# Setup the kernel
mod = SourceModule("""
__global__ void add(float *a, float *b, float *c, float *c_map) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float val;

	val = a[idx] + b[idx];
	c[idx] = val;
	c_map[idx] = val;
}
""")
add = mod.get_function("add")

# Memory allocation
nx = 1024
a = np.random.randn(nx).astype(np.float32)
b = np.random.randn(nx).astype(np.float32)
c = np.zeros_like(a)

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)

# Page-locked host memory allocation for zero-copy
c_map = cuda.pagelocked_zeros(nx, np.float32)

add( a_gpu, b_gpu, cuda.Out(c), cuda.Out(c_map), block=(256,1,1), grid=(4,1) )
assert( np.linalg.norm( (a+b)-c ) == 0 )
assert( np.linalg.norm( (a+b)-c_map ) == 0 )
