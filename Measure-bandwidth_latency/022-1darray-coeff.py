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
	nx = 120*(1024**2)/4
	nloop = 30

	# memory allocate
	fw_gpu = cuda.to_device( np.zeros(nx,'f') )
	f = np.random.randn(nx).astype(np.float32)
	fr_gpu_list = []
	for i in range(nloop): fr_gpu_list.append( cuda.to_device(f) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	for i in range(nloop):
		args = 'float *fr00'
		body = 'fr00[idx]'
		for j in range(1, i+1): 
			args += ', float *fr%.2d' % j
			body += ' + %g*fr%.2d[idx]' % (j/37., j)
		kernels += kernel_template.replace('NAME','func%.2d'%i).replace('ARGS',args).replace('BODY',body)
	print kernels

	mod = SourceModule( kernels )
	kern_list = []
	for i in range(nloop): kern_list.append( mod.get_function("func%.2d"%i) )

	tpb = (512,1,1)
	bpg = (nx/512,1)

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	exec_time = np.zeros(nloop, dtype=np.float64)
	for k in range(10):
		for i in range(nloop): 
			fr_gpus = 'fr_gpu_list[0]'
			for j in range(1, i+1): 
				fr_gpus += ', fr_gpu_list[%d]' % j
			cmd = 'kern_list[%d](fw_gpu, %s, block=tpb, grid=bpg)' % (i, fr_gpus)
			#print cmd

			start.record()
			eval(cmd) 
			stop.record()
			stop.synchronize()
			exec_time[i] += stop.time_since(start)	# ms

	exec_time[:] /= 10

	print exec_time
	dt = exec_time[1:] - exec_time[:-1]
	print dt
	
	import matplotlib.pyplot as pl
	pl.plot(dt, 'o-')
	pl.show()
