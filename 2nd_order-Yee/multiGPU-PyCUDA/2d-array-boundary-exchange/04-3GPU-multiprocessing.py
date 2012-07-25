#!/usr/bin/env python

import pycuda.driver as cuda
import numpy as np
from multiprocessing import Process

nof = np.nbytes['float32']	# nbytes of float

def allocate_gpu(device,arr):
	ctx = cuda.Device(device).make_context()
	arr_gpu = cuda.to_device(arr)
	ctx.pop()
	return arr_gpu


def print_gpu(device,arr,arr_gpu):
	ctx = cuda.Device(device).make_context()
	print cuda.from_device(arr_gpu,arr.shape,arr.dtype)
	ctx.pop()


def exchange(nx, ny, a_gpu, b_gpu, dev1, dev2):
	ctx1 = cuda.Device(dev1).make_context()
	a = cuda.from_device(int(a_gpu)+(nx-2)*ny*nof, (ny,), np.float32)
	ctx1.pop()

	ctx2 = cuda.Device(dev2).make_context()
	cuda.memcpy_htod(int(b_gpu), a)
	b = cuda.from_device(int(b_gpu)+ny*nof, (ny,), np.float32)
	ctx2.pop()

	ctx1 = cuda.Device(dev1).make_context()
	cuda.memcpy_htod_async(int(a_gpu)+(nx-1)*ny*nof, b)
	ctx1.pop()
	

if __name__ == '__main__':
	cuda.init()
	nx, ny = 6, 5
	a = np.zeros((nx,ny),'f')
	b, c = np.zeros_like(a), np.zeros_like(a)

	a[-2,:] = 1.5
	b[1,:] = 2.0
	b[-2,:] = 2.5
	c[1,:] = 3.0

	a_gpu = allocate_gpu(0,a)
	ps0 = Process(target=allocate_gpu,args=(0,a))
	#ps1 = Process(target=allocate_gpu,args=(1,b))
	#ps2 = Process(target=allocate_gpu,args=(2,c))
	ps0.start()
	#b_gpu = ps1.start()
	#c_gpu = ps2.start()
	#ps0.join()
	#ps1.join()
	#ps2.join()
	
	'''
	ps3 = Process(target=allocate_gpu,args=(0,a))
	ps4 = Process(target=allocate_gpu,args=(1,a))
	ps5 = Process(target=allocate_gpu,args=(2,a))
	print a
	ctx0 = cuda.Device(0).make_context()
	a_gpu = cuda.to_device(a)
	ctx0.pop()

	a2 = np.zeros_like(a)
	ctx1 = cuda.Device(0).make_context()
	cuda.memcpy_dtoh(a2, a_gpu)
	ctx1.pop()
	print a2

	
	ctx1 = cuda.Device(1).make_context()
	b_gpu = cuda.to_device(b)
	ctx1.pop()

	ctx2 = cuda.Device(2).make_context()
	c_gpu = cuda.to_device(c)
	ctx2.pop()
	
	a2, b2, c2 = np.zeros_like(a), np.zeros_like(a), np.zeros_like(a)
	print_abc_gpu(a2,b2,c2,a_gpu,b_gpu,c_gpu)


	ctx0 = cuda.Device(0).make_context()
	cuda.memcpy_dtoh(a, a_gpu)
	ctx0.pop()

	ctx1 = cuda.Device(1).make_context()
	cuda.memcpy_dtoh(b, b_gpu)
	ctx1.pop()

	ctx2 = cuda.Device(2).make_context()
	cuda.memcpy_dtoh(c, c_gpu)
	ctx2.pop()

	for i in xrange(a.shape[1]):
		print a[:,i], '\t', b[:,i], '\t', c[:,i]

	print_abc_gpu(a2,b2,c2,a_gpu,b_gpu,c_gpu)
	print '\nAfter exchange'
	exchange(nx, ny, a_gpu, b_gpu, 0, 1)
	exchange(nx, ny, b_gpu, c_gpu, 1, 2)
	print_abc_gpu(a2,b2,c2,a_gpu,b_gpu,c_gpu)
	'''
