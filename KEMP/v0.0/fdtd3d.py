#!/usr/bin/env python

import pyopencl as cl
import numpy as np


class EMField3dGpu:
	def __init__(s, context, queue, nx, ny, nz, dtype=np.float32):
		s.context = context
		s.queue = queue
		s.nx = nx
		s.ny = ny
		s.nz = nz
		s.dtype = dtype

		mf = cl.mem_flags
		f = np.zeros((s.nx, s.ny, s.nz), dtype=s.dtype)
		cf = np.ones_like(f) * 0.5

		s.ehs = s.ex, s.ey, s.ez, s.hx, s.hy, s.hz = [cl.Buffer(s.context, mf.READ_WRITE, f.nbytes) for i in range(6)]
		s.ces = [cl.Buffer(s.context, mf.READ_ONLY, cf.nbytes) for i in range(3)]

		for eh in s.ehs: cl.enqueue_write_buffer(queue, eh, f) 
		for ce in s.ces: cl.enqueue_write_buffer(queue, ce, cf) 

		del f, cf


	def __getitem__(s, str_f):
		return s.__dict__[str_f]



class Fdtd3dSingleGpu:
	def __init__(s, emfield, global_work_size, local_work_size=256):
		s.emf = emfield
		s.gs = global_work_size
		s.ls = local_work_size

		s.context = s.emf.context
		s.queue = s.emf.queue
		s.nx = s.emf.nx
		s.ny = s.emf.ny
		s.nz = s.emf.nz

		ksrc = utils.get_ksrc('./fdtd3d_main.cl', s.nx, s.ny, s.nz, s.ls)
		s.program = cl.Program(s.context, ksrc).build()

		s.h_args = s.emf.ehs
		s.e_args = s.emf.ehs + s.emf.ces


	def update_h(s):
		s.program.update_h(s.queue, (s.gs,), (s.ls,), *s.h_args)


	def update_e(s):
		s.program.update_e(s.queue, (s.gs,), (s.ls,), *s.e_args)



class Fdtd3dSrcGpu:
	def __init__(s, emfield, str_f, global_work_size, local_work_size=256):
		s.emf = emfield
		s.f = s.emf[str_f]
		s.gs = global_work_size
		s.ls = local_work_size

		s.context = s.emf.context
		s.queue = s.emf.queue
		s.nx = s.emf.nx
		s.ny = s.emf.ny
		s.nz = s.emf.nz

		ksrc = utils.get_ksrc('./fdtd3d_src.cl', s.nx, s.ny, s.nz, s.ls)
		s.program = cl.Program(s.context, ksrc).build()


	def update(s, tstep):
		s.program.update(s.queue, (s.gs,), (s.ls,), np.float32(tstep), s.f)



if __name__ == '__main__':
	#nx, ny, nz = 240, 256, 256		# 540 MB
	nx, ny, nz = 512, 480, 480		# 3.96 GB
	#nx, ny, nz = 480, 480, 480		# 3.71 GB
	tmax, tgap = 200, 10
	gpu_id = 0


	import utils
	gpu_devices = utils.get_gpu_devices()
	utils.print_gpu_info(gpu_devices)
	context, queues = utils.create_context_queues(gpu_devices)
	queue = queues[gpu_id]

	gs = utils.get_optimal_global_work_size( gpu_devices[gpu_id] )
	emf = EMField3dGpu(context, queue, nx, ny, nz)
	fdtd = Fdtd3dSingleGpu(emf, gs)
	src = Fdtd3dSrcGpu(emf, 'ez', gs)


	# Plot
	import matplotlib.pyplot as plt
	plt.ion()

	f = np.ones((nx, ny, nz), 'f')
	imsh = plt.imshow(f[:,:,nz/2].T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
	plt.colorbar()


	# Main loop
	import sys
	from datetime import datetime
	t0 = datetime.now()

	for tstep in xrange(1, tmax+1):
		fdtd.update_h()
		fdtd.update_e()
		src.update(tstep)

		if tstep % tgap == 0:
			print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
			sys.stdout.flush()

			cl.enqueue_read_buffer(queue, emf['ez'], f)
			imsh.set_array(f[:,:,nz/2].T**2 )
			plt.draw()

	print('')
