#!/usr/bin/env python

kernel_template = """
__global__ void NAME(float *fw, float *fr1, float *fr2) {
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
	fr1_gpu = cuda.to_device( np.random.randn(nx).astype(np.float32) )
	fr2_gpu = cuda.to_device( np.random.randn(nx).astype(np.float32) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	body = 'fr1[idx] + fr2[idx]'
	for i in range(nloop):
		if( i>0 ): body += ' + fr1[idx] + fr2[idx]'
		kernels += kernel_template.replace('NAME','func%.2d'%i).replace('BODY',body)
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
			cmd = 'kern_list[%d](fw_gpu, fr1_gpu, fr2_gpu, block=tpb, grid=bpg)' % (i)
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
