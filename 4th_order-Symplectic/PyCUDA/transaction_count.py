#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels =""" 
__global__ void func(int tn, float *a, float *b, float *c) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	a[idx] = tn*1.387689*b[idx];
	//b[idx] += tn*0.877989;
	//c[idx] += tn*5.312869;
}
"""


if __name__ == '__main__':
	#nx, ny, nz = 16, 8, 4
	nx, ny, nz = 32, 32, 16

	print 'dim (%d, %d, %d)' % (nx, ny, nz)

	# memory allocate
	#f = np.zeros((nx,ny,nz), 'f', order='F')
	f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz),order='F')

	a_gpu = cuda.to_device(f)
	b_gpu = cuda.to_device(f)
	c_gpu = cuda.to_device(f)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	func = mod.get_function("func")

	tpb = 256
	bpg = (nx*ny*nz)/tpb

	Db = (tpb,1,1)
	Dg = (bpg, 1)
	print Db, Dg

	func.prepare("iPPP", block=Db)

	# main loop
	for tn in xrange(1, 10+1):
		func.prepared_call(Dg, np.int32(tn), a_gpu, b_gpu, c_gpu);
