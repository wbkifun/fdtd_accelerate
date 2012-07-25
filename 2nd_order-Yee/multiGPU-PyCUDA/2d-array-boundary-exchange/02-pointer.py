#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


def print_abc_gpu(a,b,c,a_gpu,b_gpu,c_gpu):
	cuda.memcpy_dtoh(a, a_gpu)
	cuda.memcpy_dtoh(b, b_gpu)
	cuda.memcpy_dtoh(c, c_gpu)
	for i in xrange(a.shape[1]):
		print a[:,i], '\t', b[:,i], '\t', c[:,i]


def exchange(nx, ny, a_gpu, b_gpu):
	nof = np.nbytes['float32']	# nbytes of float
	cuda.memcpy_htod(int(b_gpu), cuda.from_device(int(a_gpu)+(nx-2)*ny*nof, (ny,), np.float32))
	cuda.memcpy_htod_async(int(a_gpu)+(nx-1)*ny*nof, cuda.from_device(int(b_gpu)+ny*nof, (ny,), np.float32))
	

if __name__ == '__main__':
	nx, ny = 6, 5
	a = np.zeros((nx,ny),'f')
	b, c = np.zeros_like(a), np.zeros_like(a)

	a[-2,:] = 1.5
	b[1,:] = 2.0
	b[-2,:] = 2.5
	c[1,:] = 3.0

	a_gpu = cuda.to_device(a)
	b_gpu = cuda.to_device(b)
	c_gpu = cuda.to_device(c)

	a2, b2, c2 = np.zeros_like(a), np.zeros_like(a), np.zeros_like(a)
	print_abc_gpu(a2,b2,c2,a_gpu,b_gpu,c_gpu)
	print '\nAfter exchange'
	exchange(nx, ny, a_gpu, b_gpu)
	exchange(nx, ny, b_gpu, c_gpu)
	print_abc_gpu(a2,b2,c2,a_gpu,b_gpu,c_gpu)
