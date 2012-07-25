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
	size = 100	# MBtye
	nx = size*(1024**2)/4
	nloop = 30

	# memory allocate
	fw_gpu = cuda.to_device( np.zeros(nx,'f') )
	f = np.random.randn(nx).astype(np.float32)
	fr_gpu_list = []
	for i in range(nloop): fr_gpu_list.append( cuda.to_device(f) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	args = 'float *fr00'
	body = '0.543*fr00[idx]'
	for i in range(nloop):
		if( i>0 ): 
			args += ', float *fr%.2d' % i
			body += ' + %1.3f*fr%.2d[idx]' % (np.random.ranf(), i)
		kernels += kernel_template.replace('NAME','func%.2d'%i).replace('ARGS',args).replace('BODY',body)
	print kernels

	mod = SourceModule( kernels )
	kern_list = []
	for i in range(nloop): kern_list.append( mod.get_function("func%.2d"%i) )

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	exec_time = np.zeros(nloop, dtype=np.float64)
	for k in range(10):
		for i in range(nloop): 
			fr_gpus = 'fr_gpu_list[0]'
			for j in range(1, i+1): 
				fr_gpus += ', fr_gpu_list[%d]' % j
			cmd = 'kern_list[%d](fw_gpu, %s, block=(512,1,1), grid=(nx/512,1))' % (i, fr_gpus)
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

	set_std = 0.04
	for j in range(nloop-1):
		print dt[j:].std()
		if( dt[j:].std() < set_std ):
			sj = j
			break
	dt_mean = dt[sj:].mean()
	print "[%.3d MByte] dt[%d:].mean()= %g" % (size, sj, dt_mean)


	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax2 = ax1.twinx()
	ax1.set_title('%.3d MByte -- std<%g, mean [%d:]'%(size, set_std, sj))
	ax1.set_xlabel('Number of Read-Access')
	ax1.set_ylabel('Time Difference [ms]', color='red')
	ax2.set_ylabel('Kernel Execution Time [ms]', color='blue')
	ax2.plot(np.arange(1,nloop+1), exec_time, 'bo')
	ax1.plot(np.arange(1.5,nloop+0.5), dt, 'ro')
	ax1.set_ylim( 0.9*dt.min(), 1.1*dt_mean )
	#ax1.axhline( dt_mean, color='red')
	#ax1.text(0.5, 1.0*dt_mean, '%g'%dt_mean, color='red', fontsize=12)
	plt.show()
