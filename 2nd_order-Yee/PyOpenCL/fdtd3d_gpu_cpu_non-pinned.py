#!/usr/bin/env python

import sys
import pyopencl as cl
import numpy as np
import utils


class Fdtd3dGpu:
	def __init__(s, nxs, ny, nz, target_device='all', print_device_info=True):
		s.gpu_devices = utils.get_gpu_devices()
		if print_device_info:
			utils.print_gpu_info(s.gpu_devices)
			utils.print_cpu_info()
		ngpu_dev = len(s.gpu_devices)

		s.context, s.queues = utils.create_context_queues(s.gpu_devices)
		s.ngpu = len(s.gpu_devices)
		s.Ls = 256
		if target_device == int:
			s.Gs = utils.get_optimal_global_work_size(s.gpu_devices[target_device])
		else:
			s.Gs = utils.get_optimal_global_work_size(s.gpu_devices[0])

		if type(nxs) == list:
			if len(nxs) == s.ngpu:
				s.nxs = nxs
				s.nx_gpu = np.array(nxs).sum()
			else:
				print('Error: len(nxs) %d is not matched with the number of target devices %d.' %(len(nxs), s.ngpu))
				sys.exit()
		elif type(nxs) == int:
			if nxs % s.ngpu == 0:
				s.nxs = [nxs/s.ngpu for i in xrange(s.ngpu)]
				s.nx_gpu = nxs
			else:
				print('Error: nxs %d is not multiple of the number of target devices %d.' %(nxs, s.ngpu))
				sys.exit()
		else:
			print('Error: nxs type %s is invalid.' %type(nxs))
			sys.exit()

		s.ny, s.nz = ny, nz
		s.check_grid_size()
		s.allocations()
		s.get_program(print_source=False)


	def check_grid_size(s):
		utils.print_nbytes('gpu (global)', s.nx_gpu, s.ny, s.nz, 9)
		for i, nx in enumerate(s.nxs):
			nbytes = utils.print_nbytes('gpu #%d' %i, nx, s.ny, s.nz, 9)
			gmem_size = s.gpu_devices[i].get_info(cl.device_info.GLOBAL_MEM_SIZE)
			if nbytes >= gmem_size:
				print('Error: The grid size %d is over the global memory %d.' %(nbytes, gmem_size))
				sys.exit()

		if s.nz % 32 != 0:
			print('Error: nz is not multiple of 32')
			sys.exit()


	def allocations(s):
		s.eh_fields_gpus = []
		s.ce_fields_gpus = []

		mf = cl.mem_flags
		for nx, queue in zip(s.nxs, s.queues):
			f = np.zeros((nx, s.ny, s.nz), 'f')
			cf = np.ones_like(f) * 0.5
			s.eh_fields_gpus.append( [cl.Buffer(s.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f) for m in range(6)] ) 
			s.ce_fields_gpus.append( [cl.Buffer(s.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf) for m in range(3)] )
			del f, cf

		s.offsets = []
		s.tmpfs = []
		for nx in s.nxs:
			s.offsets.append( (nx-1) * s.ny * s.nz * np.nbytes['float32'] )
			s.tmpfs.append( [np.zeros((s.ny, s.nz), dtype=np.float32) for m in range(2)] )


	def get_program(s, print_source=False):
		kern = open('./fdtd3d.cl').read()
		kernels = []
		s.programs = []
		for nx in s.nxs:
			kernels.append( kern.replace('NXYZ',str(nx * s.ny * s.nz)).\
					replace('NYZ',str(s.ny * s.nz)).\
					replace('NX',str(nx)).\
					replace('NY',str(s.ny)).\
					replace('NZ',str(s.nz)).\
					replace('DX',str(s.Ls)) )
			s.programs.append(cl.Program(s.context, kernels[-1]).build())

		if print_source: 
			for kernel in kernels: print kernel


	def update_h(s):
		for program, queue, eh_fields in zip(s.programs, s.queues, s.eh_fields_gpus):
			program.update_h(queue, (s.Gs,), (s.Ls,), *eh_fields)

	
	def update_e(s):
		for program, queue, eh_fields, ce_fields in zip(s.programs, s.queues, s.eh_fields_gpus, s.ce_fields_gpus):
			program.update_e(queue, (s.Gs,), (s.Ls,), *(eh_fields + ce_fields))


	def exchange_boundary_h(s):
		for queue, eh_fields, tmpf, offset in zip(s.queues, s.eh_fields_gpus, s.tmpfs, s.offsets)[:-1]:
			cl.enqueue_read_buffer(queue, eh_fields[4], tmpf[0], offset)	# hy_gpu
			cl.enqueue_read_buffer(queue, eh_fields[5], tmpf[1], offset)	# hz_gpu
		for queue, eh_fields, tmpf in zip(s.queues[1:], s.eh_fields_gpus[1:], s.tmpfs[:-1]):
			cl.enqueue_write_buffer(queue, eh_fields[4], tmpf[0])
			cl.enqueue_write_buffer(queue, eh_fields[5], tmpf[1])


	def exchange_boundary_e(s):
		for queue, eh_fields, tmpf in zip(s.queues, s.eh_fields_gpus, s.tmpfs)[1:]:
			cl.enqueue_read_buffer(queue, eh_fields[1], tmpf[0])	# ey_gpu
			cl.enqueue_read_buffer(queue, eh_fields[2], tmpf[1])	# ez_gpu
		for queue, eh_fields, tmpf, offset in zip(s.queues[:-1], s.eh_fields_gpus[:-1], s.tmpfs[1:], s.offsets[:-1]):
			cl.enqueue_write_buffer(queue, eh_fields[1], tmpf[0], offset)
			cl.enqueue_write_buffer(queue, eh_fields[2], tmpf[1], offset)



class Fdtd3dCpu:
	def __init__(s, nx, ny, nz, print_device_info=True):
		s.nx, s.ny, s.nz = nx, ny, nz
		s.check_grid_size()
		s.allocations()
		s.get_program(print_source=True)


	def check_grid_size(s):
		nbytes = utils.print_nbytes('cpu', s.nx, s.ny, s.nz, 9)
		for line in open('/proc/meminfo'):
			if 'MemTotal' in line:
				mem_size = int(line[line.find(':')+1:line.rfind('kB')]) * 1024
				break
		if nbytes >= mem_size:
			print('Error: The grid size %d is over the host memory %d.' %(nbytes, gmem_size))
			sys.exit()

		if s.nz % 4 != 0:
			print('Error: nz is not multiple of 4')
			sys.exit()


	def allocations(s):
		s.eh_fields = [np.zeros((s.nx, s.ny, s.nz), dtype=np.float32) for i in xrange(6)]
		s.ce_fields = [np.ones((s.nx, s.ny, s.nz), dtype=np.float32)*0.5 for i in xrange(3)]


	def get_program(s, print_source=False):
		kern = open('./fdtd3d.c').read()
		kernels = kern.replace('NXYZ',str(s.nx * s.ny * s.nz)).\
				replace('NYZ',str(s.ny * s.nz)).\
				replace('NXY',str(s.nx * s.ny)).\
				replace('NX',str(s.nx)).\
				replace('NY',str(s.ny)).\
				replace('NZ',str(s.nz)).\
				replace('OMP_MAX_THREADS',str(4))
		if print_source: print kernels

		wf = open('/tmp/fdtd3d.c', 'w')
		wf.write(kernels)
		wf.close()
		cmd = 'gcc -O3 -std=c99 -fpic -shared -fopenmp -msse %s -o /tmp/libfdtd3d.so' %(wf.name)
		import subprocess as sp
		proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
		stdoutdata, stderrdata = proc.communicate()
		print stdoutdata
		print stderrdata

		s.clib = np.ctypeslib.load_library('libfdtd3d', '/tmp/')
		ex = s.eh_fields[0]
		arg = np.ctypeslib.ndpointer(dtype=ex.dtype, ndim=ex.ndim, shape=ex.shape, flags='C_CONTIGUOUS, ALIGNED')
		#arg = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, shape=(s.nx, s.ny, s.nz), flags='C_CONTIGUOUS, ALIGNED')
		s.clib.update_h.argtypes = [arg for i in xrange(6)]
		s.clib.update_e.argtypes = [arg for i in xrange(9)]
		s.clib.update_h.restype = None
		s.clib.update_e.restype = None


	def update_h(s):
		s.clib.update_h(*s.eh_fields)

	
	def update_e(s):
		s.clib.update_e(*(s.eh_fields + s.ce_fields))




if __name__ == '__main__':
	nx, ny, nz = 240, 256, 256		# 540 MB
	#nx, ny, nz = 512, 480, 480		# 3.96 GB
	#nx, ny, nz = 256, 480, 960
	tmax, tgap = 200, 10

	nxs = nx*3
	s = Fdtd3dGpu(nxs, ny, nz, target_device='all')


	# Plot
	import matplotlib.pyplot as plt
	plt.ion()

	global_f = np.ones((nxs, ny), 'f')
	imsh = plt.imshow(global_f.T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
	plt.colorbar()


	# Main loop
	from datetime import datetime
	t0 = datetime.now()

	for tstep in xrange(1, tmax+1):
		s.update_h()
		s.exchange_boundary_h()
		s.update_e()
		s.exchange_boundary_e()
		s.prg.update_src(s.queues[1], (s.Gs,), (s.Ls,), np.float32(tstep), s.eh_fields_gpus[1][2])	# dev #1, ez_gpu

		if tstep % tgap == 0:
			print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
			sys.stdout.flush()

			for i, (nx, queue) in enumerate(zip(s.nxs, s.queues)):
				f = np.zeros((nx, ny, nz), 'f')
				cl.enqueue_read_buffer(queue, s.eh_fields_gpus[i][2], f)	# ez_gpu
				global_f[i*nx:(i+1)*nx,:] = f[:,:,nz/2]
			imsh.set_array( global_f.T**2 )
			plt.draw()

	print('')
