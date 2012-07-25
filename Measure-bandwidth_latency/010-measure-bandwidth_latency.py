#!/usr/bin/env python

kernel_template = """
__global__ void NAME(float *fw, ARGS) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	fw[idx] = BODY;
}
"""

kernels = """"""

import numpy as np
import sys
import pycuda.driver as cuda
import pycuda.autoinit

if __name__ == '__main__':
	nx, ny, nz = 240, 256, 256
	nloop = 15

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*(1+nloop)
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz))
	fw_gpu = cuda.to_device( np.zeros((nx,ny,nz), 'f') )
	fr_gpu_list = []
	for i in range(nloop): fr_gpu_list.append( cuda.to_device(f) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	for i in range(nloop):
		args = 'float *fr00'
		body = 'fr00[idx]'
		for j in range(1, i+1): 
			args += ', float *fr%.2d' % j
			body += ' + fr%.2d[idx]' % j
		kernels += kernel_template.replace('NAME','func%.2d'%i).replace('ARGS',args).replace('BODY',body)
	#print kernels

	mod = SourceModule( kernels )
	kern_list = []
	for i in range(nloop): kern_list.append( mod.get_function("func%.2d"%i) )

	tpb = (256,1,1)
	bpg = (nx*ny*nz/256,1)

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	for i in range(nloop): kern_list.append( mod.get_function("func%.2d"%i) )
	for i in range(nloop): 
		fr_gpus = 'fr_gpu_list[0]'
		for j in range(1, i+1): 
			fr_gpus += ', fr_gpu_list[%d]' % j

		eval('kern_list[i](fw_gpu, %s, block=tpb, grid=bpg)' % fr_gpus) 

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
