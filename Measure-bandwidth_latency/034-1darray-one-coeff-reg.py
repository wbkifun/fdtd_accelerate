#!/usr/bin/env python

kernel_template = """
__global__ void NAME(float *fw, float *fr) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float f;
	BODY
	fw[idx] = f;
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
	nloop = 50

	# memory allocate
	fw_gpu = cuda.to_device( np.zeros(nx,'f') )
	fr_gpu = cuda.to_device( np.random.randn(nx).astype(np.float32) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	body = 'f = 0.543*fr[idx];\n'
	for i in range(nloop):
		if( i>0 ): body += '\tf += %1.3f*fr[idx];\n' % (np.random.ranf())
		kernels += kernel_template.replace('NAME','func%.2d'%i).replace('BODY',body)
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
			cmd = 'kern_list[%d](fw_gpu, fr_gpu, block=(512,1,1), grid=(nx/512,1))' % i
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

	set_std = 0.003
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
	ax1.axhline( dt_mean, color='red')
	ax1.text(0.5, 1.0*dt_mean, '%g'%dt_mean, color='red', fontsize=12)
	plt.show()
