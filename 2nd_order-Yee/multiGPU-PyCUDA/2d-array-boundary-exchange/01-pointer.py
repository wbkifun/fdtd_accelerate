#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

nx, ny = 8, 5
a = np.zeros((nx,ny),'f')
b = np.zeros_like(a)
c = np.zeros_like(a)

a[-2,:] = 1.5
b[1,:] = 2.0
b[-2,:] = 2.5
c[1,:] = 3.0

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.to_device(c)

print 'a_gpu'
print cuda.from_device(a_gpu, a.shape, a.dtype)
print 'b_gpu'
print cuda.from_device(b_gpu, b.shape, b.dtype)
print 'c_gpu'
print cuda.from_device(c_gpu, c.shape, c.dtype)

print '\nAfter exchange...\n'

def exchange(nx, ny, a_gpu, b_gpu):
	nof = np.nbytes['float32']	# nbytes of float
	cuda.memcpy_htod(int(b_gpu), cuda.from_device(int(a_gpu)+(nx-2)*ny*nof, (ny,), np.float32))
	cuda.memcpy_htod_async(int(a_gpu)+(nx-1)*ny*nof, cuda.from_device(int(b_gpu)+ny*nof, (ny,), np.float32))
	
exchange(nx, ny, a_gpu, b_gpu)
exchange(nx, ny, b_gpu, c_gpu)
print 'a_gpu'
print cuda.from_device(a_gpu, a.shape, a.dtype)
print 'b_gpu'
print cuda.from_device(b_gpu, b.shape, b.dtype)
print 'c_gpu'
print cuda.from_device(c_gpu, c.shape, c.dtype)
