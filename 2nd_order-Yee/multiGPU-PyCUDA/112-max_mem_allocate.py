#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


def mem_alloc(n):
	f = np.zeros(n, 'f')
	cf = np.ones_like(f)*0.5

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device( cf )
	cey_gpu = cuda.to_device( cf )
	cez_gpu = cuda.to_device( cf )


	del f, cf
	del hx_gpu, hy_gpu, hz_gpu
	del ex_gpu, ey_gpu, ez_gpu
	del cex_gpu, cey_gpu, cez_gpu


for n in xrange( 3.9*1024**3, 4*1024**3, 1.024*1024**2):
	n /= 4*9
	print 'n= %d' % (n)
	total_bytes = n*4*9
	print 'mem %f MB' % ( total_bytes/(1024**2) )

	mem_alloc(n)
