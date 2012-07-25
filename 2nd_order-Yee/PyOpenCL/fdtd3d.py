#!/usr/bin/env python

import sys
import pyopencl as cl
import numpy as np
import utils


class Fdtd3d:
	def __init__(s, nxs, ny, nz, target_device='all', print_verbose=True):
		s.print_verbose = print_verbose
		s.gpu_devices = utils.get_gpu_devices(s.print_verbose)
		if s.print_verbose:
			utils.print_gpu_info(s.gpu_devices)
			utils.print_cpu_info()
		ngpu_dev = len(s.gpu_devices)

		s.lsize = 256
		s.gsizes = []
		s.nnx = 1
		s.ngpu = ngpu_dev
		s.context, s.queues = utils.create_context_queues(s.gpu_devices)
		td = target_device
		if ngpu_dev > 0:
			for device in s.gpu_devices:
				s.gsizes.append( utils.get_optimal_global_work_size(device) )

			if td == 'cpu':
				s.ngpu = 0
				target_str = 'CPU'
			elif td in ['gpu%d' % i for i in range(ngpu_dev)]:
				s.ngpu = 1
				gpu_num = int(td.strip('gpu'))
				s.gsizes = [ s.gsizes[gpu_num] ]
				s.gpu_devices = [ s.gpu_devices[gpu_num] ]
				s.context, s.queues = utils.create_context_queues(s.gpu_devices)
				target_str = 'Single GPU #%d' % gpu_num
			elif td in ['gpu']:
				s.nnx = ngpu_dev
				target_str = '%d GPUs' % s.ngpu
			elif td in ['all']:
				s.nnx = ngpu_dev + 1
				target_str = 'CPU + %d GPUs' % s.ngpu
			else:
				print('Error: Invalid target_device option.')
				print('      Possible options: %s' %(['all', 'cpu', 'gpu'] +  ['gpu%d' % i for i in range(ngpu_dev)]))
				sys.exit()
		else:
			if td in ['all', 'cpu']:
				s.nnx = 1
				s.ngpu = 0
				target_str = 'CPU'
			else:
				print('Error: Invalid target_device option.')
				print('      There are no GPU devices.')
				print('      Possible options: %s' %(['all', 'cpu']))
				sys.exit()

		if type(nxs) == list:
			if len(nxs) == s.nnx:
				s.nxs = nxs
				s.nx_total = np.array(nxs).sum()
			else:
				print('Error: len(nxs) %d is not matched with the number of target devices %d.' %(len(nxs), s.nnx))
				sys.exit()
		elif type(nxs) == int:
			s.nx_total = nxs
			if s.nnx == 1:
				s.nxs = [nxs]
			else:
				#s.nxs = utils.get_optimal_nxs()
				s.nxs = [nxs/s.ngpu for i in xrange(s.ngpu)]
		else:
			print('Error: nxs type %s is invalid.' % type(nxs))
			print('      Possible types: %s' %(['list', 'int']))
			sys.exit()

		if s.print_verbose:
			print('Target Device : %s' % target_str)
			print('s.nnx = %d' % s.nnx)
			print('s.ngpu = %d' % s.ngpu)
			print('s.nxs = %s' % s.nxs)
			print('')

		s.ny, s.nz = ny, nz
		s.check_grid_size()
		s.allocations()
		s.get_program(print_ksource=False)
		s.prepare_updates()


	def check_grid_size(s):
		nbytes = s.nx_total * s.ny * s.nz * np.nbytes['float32'] * 9
		if s.print_verbose:
			print('Total (%d, %d, %d) ' % (s.nx_total, s.ny, s.nz)),
			print('%1.2f %s' % utils.get_nbytes_unit(nbytes))
		for i, nx in enumerate(s.nxs):
			nbytes = nx * s.ny * s.nz * np.nbytes['float32'] * 9
			if i < s.ngpu:
				mem_size = s.gpu_devices[i].get_info(cl.device_info.GLOBAL_MEM_SIZE)
				head_str = 'GPU #%d' % i
			else:
				for line in open('/proc/meminfo'):
					if 'MemTotal' in line:
						mem_size = int(line[line.find(':')+1:line.rfind('kB')]) * 1024
						break
				head_str = 'CPU'
			if s.print_verbose:
				print('%s (%d, %d, %d) ' % (head_str, nx, s.ny, s.nz)),
				print('%1.2f %s' % utils.get_nbytes_unit(nbytes))

			if nbytes >= mem_size:
				print('Error: The required memory size %d is over the global memory %d.' %(nbytes, gmem_size))
				sys.exit()
		if s.print_verbose: print('')

		if s.nz % 32 != 0:
			print('Error: nz is not multiple of 32')
			sys.exit()


	def allocations(s):
		s.eh_fieldss = []
		s.ce_fieldss = []
		mf = cl.mem_flags
		for i, nx in enumerate(s.nxs):
			f = np.zeros((nx, s.ny, s.nz), 'f')
			cf = np.ones_like(f) * 0.5

			if i < s.ngpu:
				s.eh_fieldss.append( [cl.Buffer(s.context, mf.READ_WRITE, f.nbytes) for m in range(6)] ) 
				s.ce_fieldss.append( [cl.Buffer(s.context, mf.READ_ONLY, cf.nbytes) for m in range(3)] )
				for eh_field in s.eh_fieldss[-1]:
					cl.enqueue_write_buffer(s.queues[i], eh_field, f) 
				for ce_field in s.ce_fieldss[-1]:
					cl.enqueue_write_buffer(s.queues[i], ce_field, cf) 
			else:
				s.eh_fieldss.append( [f.copy() for i in xrange(6)] )
				s.ce_fieldss.append( [cf.copy() for i in xrange(3)] )

			del f, cf

		s.offsets = []
		s.tmpfs = []
		for nx in s.nxs:
			s.offsets.append( (nx-1) * s.ny * s.nz * np.nbytes['float32'] )
			s.tmpfs.append( [np.zeros((s.ny, s.nz), dtype=np.float32) for m in range(2)] )


	def get_program(s, print_ksource=False):
		s.programs = []

		ksrc_gpu = open('./fdtd3d.cl').read()
		ksrc_cpu = open('./fdtd3d.c').read()
		for i, nx in enumerate(s.nxs):
			if i < s.ngpu: ksrc = ksrc_gpu
			else: ksrc = ksrc_cpu

			ksource = ksrc.replace('NXYZ',str(nx * s.ny * s.nz)).\
					replace('NYZ',str(s.ny * s.nz)).\
					replace('NXY',str(nx * s.ny)).\
					replace('NX',str(nx)).\
					replace('NY',str(s.ny)).\
					replace('NZ',str(s.nz)).\
					replace('DX',str(s.lsize)).\
					replace('OMP_MAX_THREADS',str(4))
						
			if i < s.ngpu:
				s.programs.append(cl.Program(s.context, ksource).build())
			else:
				wf = open('/tmp/fdtd3d.c', 'w')
				wf.write(ksource)
				wf.close()
				cmd = 'gcc -O3 -std=c99 -fpic -shared -fopenmp -msse %s -o /tmp/libfdtd3d.so' %(wf.name)
				import subprocess as sp
				proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
				stdoutdata, stderrdata = proc.communicate()
				if stdoutdata != '' or stderrdata != '':
					print('stdout :\n%s\nstderr :\n %s\n' % (stdoutdata, stderrdata))

				prg = np.ctypeslib.load_library('libfdtd3d', '/tmp/')
				carg = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, shape=(nx, s.ny, s.nz), flags='C_CONTIGUOUS, ALIGNED')
				prg.update_h.argtypes = [carg for i in xrange(6)]
				prg.update_e.argtypes = [carg for i in xrange(9)]
				prg.update_h.restype = None
				prg.update_e.restype = None
				s.programs.append(prg)

			if print_ksource: print ksource


	def prepare_updates(s):
		s.update_h_funcs = []
		s.update_h_args = []
		for i, (program, eh_fields) in enumerate(zip(s.programs, s.eh_fieldss)):
			s.update_h_funcs.append(program.update_h)
			if i < s.ngpu:
				s.update_h_args.append([s.queues[i], (s.gsizes[i],), (s.lsize,)] + eh_fields)
			else:
				s.update_h_args.append(eh_fields)

		s.update_e_funcs = []
		s.update_e_args = []
		for i, (program, eh_fields, ce_fields) in enumerate(zip(s.programs, s.eh_fieldss, s.ce_fieldss)):
			s.update_e_funcs.append(program.update_e)
			if i < s.ngpu:
				s.update_e_args.append([s.queues[i], (s.gsizes[i],), (s.lsize,)] + eh_fields + ce_fields)
			else:
				s.update_e_args.append(eh_fields + ce_fields)


	def update_h(s):
		for func, args in zip(s.update_h_funcs, s.update_h_args):
			func(*args)
		
	
	def update_e(s):
		for func, args in zip(s.update_e_funcs, s.update_e_args):
			func(*args)


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



if __name__ == '__main__':
	nx, ny, nz = 240, 256, 256		# 540 MB
	#nx, ny, nz = 512, 480, 480		# 3.96 GB
	#nx, ny, nz = 256, 480, 960
	tmax, tgap = 200, 10

	nxs = nx*3
	s = Fdtd3d(nxs, ny, nz, target_device='gpu')


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
		s.prg.update_src(s.queues[1], (s.gsizes,), (s.lsize,), np.float32(tstep), s.eh_fields_gpus[1][2])	# dev #1, ez_gpu

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
