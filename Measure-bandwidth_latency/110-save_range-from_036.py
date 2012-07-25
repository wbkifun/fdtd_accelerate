#!/usr/bin/env python

kernel_template = """
__global__ void NAME(float *fw, float *fr) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float f;
	BODY
	fw[idx] = f;
}
"""

import numpy as np
import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
from datetime import datetime


def time_device_array_read_in_kernel(mb_size, nloop):
	kernels = """"""
	nx = mb_size*(1024**2)/4

	# memory allocate
	fw_gpu = cuda.to_device( np.zeros(nx,'f') )
	fr_gpu = cuda.to_device( np.random.randn(nx).astype(np.float32) )
	exec_time = np.zeros(nloop, dtype=np.float64)

	# prepare kernels
	body = 'f = 0.543*fr[idx];\n\t__syncthreads();\n'
	for i in range(nloop):
		if( i>0 ): body += '\tf += %1.5f*fr[idx];\n\t__syncthreads();\n' % (np.random.ranf())
		kernels += kernel_template.replace('NAME','func%.2d'%i).replace('BODY',body)
	#print kernels

	mod = SourceModule( kernels )
	kern_list = []
	for i in range(nloop): kern_list.append( mod.get_function("func%.2d"%i) )

	tpb = (512,1,1)
	bpg = (nx/512,1)

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	for k in range(10):
		for i in range(nloop): 
			cmd = 'kern_list[%d](fw_gpu, fr_gpu, block=tpb, grid=bpg)' % (i)
			#print cmd
			start.record()
			eval(cmd) 
			stop.record()
			stop.synchronize()
			exec_time[i] += stop.time_since(start)	# ms
	exec_time[:] /= 10
	np.save('./110-time_datas-npy/%.3dMByte.npy' % mb_size, exec_time)


if __name__ == '__main__':
	nsize = 127
	dt = np.zeros(nsize,'f')

	t1 = datetime.now()
	for i in range(1,nsize+1):
		dt[i] = time_device_array_read_in_kernel(mb_size=i, nloop=50)

		print "[",datetime.now()-t1,"]","size= %.3d MBytes (%d %%)\r" % (i, float(i)/nsize*100),
		sys.stdout.flush()
