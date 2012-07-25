import numpy as np
from kemp.fdtd3d import gpu, cpu
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


mpi_shape = (2,3,1)
pbc = 'x'
rank0_is_node = 'False'


class BufferFields(cpu.Fields):
	def __init__(s, gpu_fields_list, position, target_rank, nx, ny, nz, dtype, coeff_use, use_cpu_core):
		cpu.Fields.__init__(nx, ny, nz, dtype, coeff_use, use_cpu_core)
		s.pos = position
		s.rank = target_rank

		# get, set from gpus
		fs = gpu_fields_list
		ny, nz = fs[0].ny, fs[0].nz

		if s.pos == 'x+':
			f = fs[-1]
			s.getfs = [gpu.GetFields(f, ['hy', 'hz'], (f.nx-1, 0, 0), (f.nx-1, ny-1, nz-1))]
			s.setfs = [gpu.SetFields(f, ['ey', 'ez'], (f.nx-1, 0, 0), (f.nx-1, ny-1, nz-1))]
			s.send_tag, s.recv_tag = 11, 12

		elif s.pos == 'x-':
			f = fs[0]
			s.getfs = [gpu.GetFields(f, ['ey', 'ez'], (1, 0, 0), (1, ny-1, nz-1))]
			s.setfs = [gpu.SetFields(f, ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1))]
			s.send_tag, s.recv_tag = 12, 11

		elif s.pos == 'y+':
			s.getfs = [gpu.GetFields(fields, ['hx', 'hz'], (0, ny-2, 0), (f.nx-1, ny-2, nz-1)) for f in fs]
			s.setfs = [gpu.SetFields(fields, ['ex', 'ez'], (0, ny-1, 0), (f.nx-1, ny-1, nz-1)) for f in fs]
			s.send_tag, s.recv_tag = 21, 22
		
		elif s.pos == 'y-':
			s.getfs = [gpu.GetFields(fields, ['ex', 'ez'], (0, 1, 0), (f.nx-1, 1, nz-1)) for f in fs]
			s.setfs = [gpu.SetFields(fields, ['hx', 'hz'], (0, 0, 0), (f.nx-1, 0, nz-1)) for f in fs]
			s.send_tag, s.recv_tag = 22, 21
		
		elif s.pos == 'z+':
			s.getfs = [gpu.GetFields(fields, ['hx', 'hy'], (0, 0, nz-2), (f.nx-1, ny-1, nz-2)) for f in fs]
			s.setfs = [gpu.SetFields(fields, ['ex', 'ey'], (0, 0, nz-1), (f.nx-1, ny-1, nz-1)) for f in fs]
			s.send_tag, s.recv_tag = 31, 32
		
		elif s.pos == 'z-':
			s.getfs = [gpu.GetFields(fields, ['ex', 'ey'], (0, 0, 1), (f.nx-1, ny-1, 1)) for f in fs]
			s.setfs = [gpu.SetFields(fields, ['hx', 'hy'], (0, 0, 0), (f.nx-1, ny-1, 0)) for f in fs]
			s.send_tag, s.recv_tag = 32, 31


		# slice for one-to-many exchange
		if s.pos in ['x+', 'x-']:
			idxs = [0, f.nx]

		elif s.pos in ['y+', 'y-', 'z+', 'z-']:
			idxs = [0]
			for i, nx in [f.nx for f in fs]:
				idxs.append(idxs[i] + nx - i)

		s.sls = [ slice(idxs[i], idxs[i+1]) for i, idx in enumerate(idxs[:-1])) ]

		if s.pos == 'x+': s.to_gpu_e_idx = 0
		else: s.to_gpu_e_idx = 1

		
		# requests for persistent mpi communications
		if '-' in s.pos:
			s.req_send = comm.Send_init(s.ey[1,:,:], dest=s.rank, tag=s.send_tag)
			s.req_send = comm.Send_init(s.ez[1,:,:], dest=s.rank, tag=s.send_tag)
			s.req_recv = comm.Recv_init(s.hy[0,:,:], dest=s.rank, tag=s.recv_tag)
			s.req_recv = comm.Recv_init(s.hz[0,:,:], dest=s.rank, tag=s.recv_tag)


		# arguments for update
		if '-' in s.pos:
			s.e_args_pre = s.e_args[:3] + [np.int32(ny*nz), np.int32(2*ny*nz)] + s.e_args[5:]
			s.e_args_mid = s.e_args[:3] + [np.int32(2*ny*nz), np.int32((nx*ny-1)*nz)] + s.e_args[5:]
			s.e_args_post = s.e_args[:3] + [np.int32(0), np.int32(ny*nz)] + s.e_args[5:]
		elif '+' in s.pos:
			s.h_args_pre = s.h_args[:3] + [np.int32((nx-2)*ny*nz), np.int32((nx-1)*ny*nz)] + s.h_args[5:]
			s.h_args_mid = s.h_args[:3] + [np.int32(nz), np.int32((nx-2)*ny*nz)] + s.h_args[5:]
			s.h_args_post = s.h_args[:3] + [np.int32((nx-1)*ny*nz), np.int32(nx*ny*nz)] + s.h_args[5:]



	def from_gpu_e(s):
		for getf, sl in zip(s.getfs, s.sls):
			getf.get_event().wait()
			s.ey[-1,sl,:] = getf.split_fs[0]
			s.ez[-1,sl,:] = getf.split_fs[1]

	
	def from_gpu_h(s):
		for getf, sl in zip(s.getfs, s.sls):
			getf.get_event().wait()
			s.hy[0,sl,:] = getf.split_fs[0]
			s.hz[0,sl,:] = getf.split_fs[1]


	def to_gpu_e(s):
		for setf, sl in zip(s.setfs, s.sls):
			setf.set_fields( np.concatenate(s.ey[s.to_gpu_e_idx,sl,:], s.ez[s.to_gpu_e_idx,sl,:]) )


	def to_gpu_h(s):
		for setf, sl in zip(s.setfs, s.sls):
			setf.set_fields( np.concatenate(s.hy[1,sl,:], s.hz[1,sl,:]) )


	def send_e_start(s):
		s.qtask.enqueue(set_fields, 

	def pre_update_e(s):

	def mid_update_e(s):

	def post_update_e(s):

	def update_e(s):
		s.qtask.enqueue(s.program.update_e, *(s.nss.e_args)
		for e_func in e_func_list:
	


class ExchangeMpi:
	def __init__(s, fields_list, nx_cpu, nx, ny, nz):
		cuse = fields_list[0].coeff_use
		s.bfs = {}
		s.bfs['x+'] = BufferFields(nx_cpu, ny, nz, cuse, 0)
		s.bfs['x-'] = BufferFields(3, ny, nz, cuse, 1)
		s.bfs['y+'] = BufferFields(3, nx, nz, cuse, 1)
		s.bfs['y-'] = BufferFields(3, nx, nz, cuse, 1)
		s.bfs['z+'] = BufferFields(3, nx, ny, cuse, 1)
		s.bfs['z-'] = BufferFields(3, nx, ny, cuse, 1) 


	def update_e(s):
		for pos, bf in s.bfs.items():
			if pos == 'x+':
				getf = gpu.GetFields(s.fields_list[-1], ['hy', 'hz'], (nx-2, 0, 0), (nx-2, ny-1, nz-1))
				setf = cpu.SetFields(s.bfs[pos], ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1))
				setf.set_fields(getf.get_fields(), [getf.get_event()])
				s.bfs[pos].qtask.enqueue(s.bfs[pos].program.update_e, *s.bfs[pos].e_args)

			elif pos == 'y+':
				getfs = [gpu.GetFields(fields, ['hx', 'hz'], (nx-2, 0, 0), (nx-2, ny-1, nz-1)) for fields in s.fields_list]
				setf = cpu.SetFields(s.bfs[pos], ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1))
				setf.set_fields(getf.get_fields(), [getf.get_event()])
				s.bfs[pos].qtask.enqueue(s.bfs[pos].program.update_e, *s.bfs[pos].e_args)


	def update_h(s):



def get_pts_for_boundary(axis, nx, ny, nz):
	pt0 = {
			'x': {'-':(0, 0, 0), '+':(nx-1, 0, 0)},
			'y': {'-':(0, 0, 0), '+':(0, ny-1, 0)},
			'z': {'-':(0, 0, 0), '+':(0, 0, nz-1)} }[axis]
	pt1 = {
			'x': {'-':(0, ny-1, nz-1), '+':(nx-1, ny-1, nz-1)},
			'y': {'-':(nx-1, 0, nz-1), '+':(nx-1, ny-1, nz-1)},
			'z': {'-':(nx-1, ny-1, 0), '+':(nx-1, ny-1, nz-1)} }[axis]

	return pt0, pt1



class ExchangeInternal:
	def __init__(s, node_list, axis):
		emf_list = fields_list

		e_strfs = {'x':['ey','ez'], 'y':['ex','ez'], 'z':['ex','ey']}[axis]
		h_strfs = {'x':['hy','hz'], 'y':['hx','hz'], 'z':['hx','hy']}[axis]
		pts = [get_pts_for_boundary(axis, *emf.ns) for emf in emf_list]

		s.e_getfs, s.h_getfs = [], []
		s.e_setfs, s.h_setfs = [], []
		s.gpu, s.cpu = gpu, cpu

		for emf, (pt0, pt1) in zip(emf_list, pts)[1:]:
			s.e_getfs.append( getattr(s, emf.device_type).GetFields(emf, e_strfs, pt0['-'], pt1['-']) )
			s.h_setfs.append( getattr(s, emf.device_type).SetFields(emf, h_strfs, pt0['-'], pt1['-'], np.ndarray) )

		for emf, (pt0, pt1) in zip(emf_list, pts)[:-1]:
			s.h_getfs.append( getattr(s, emf.device_type).GetFields(emf, h_strfs, pt0['+'], pt1['+']) )
			s.e_setfs.append( getattr(s, emf.device_type).SetFields(emf, e_strfs, pt0['+'], pt1['+'], np.ndarray) )



	s.update = s.update_e_head	
	s.update = s.update_e_body	
	s.update = s.update_e_tail	


	def update_e_head(s):
		s.recv.Start()
		s.recv.Wait()
		s.setf.set_fields(s.e_fhost_recv)


	def update_e_body(s):
		s.e_recv.Start()
		s.e_getf.get_event().wait()
		s.e_send.Start()

		s.e_recv.Wait()
		s.e_setf.set_fields(s.e_fhost_recv)
		s.e_send.Wait()

	def update_e_body(s):
		f_list = [s.e_recv.Start, s.e_getf.get_event().wait, s.e_send.Start]
		arg_list = [(), (), ()]
		ff = zip(f_list, arg_list)
		for f, arg in ff:
			f(*arg)

		s.e_recv.Wait()
		s.e_setf.set_fields(s.e_fhost_recv)
		s.e_send.Wait()


		start_list = []
		wait_list = []

		[s.e_recv.Start], # recv
		[s.e_getf.get_event().wait, s.e_send.Start], # send

		[s.e_recv.Wait, s.e_setf.set_fields], # recv
		[s.e_send.Wait]

		] # send

		if 'r' in sr:
			start_list.append(  )
			wait_list.append( )
		if 's' in sr:
			start_list.append(  )
			wait_list.append( )

		
			f_list.insert(-1, ff_list[1])
			f_list = f_list + [ff_list[3]]
		fin_f_list = f_list[0,1,2,3]
			

		elif I = recv:
			f_list = ff_list[0] + ff_list[2]
		else:
			f_list = ff_list[0]+ ff_list[1]+ ff_list[2]+ ff_list[3]


		"""
		s.e_recv.Start()
		#s.e_getf.get_event().wait()
		#s.e_send.Start()

		s.e_recv.Wait()
		s.e_setf.set_fields(s.e_fhost_recv)
		#s.e_send.Wait()
		"""

	def update_e_tail(s):
		s.e_getf.get_event().wait()
		s.e_send.Start()
		s.e_send.Wait()









	def update_h_head(s):
		s.h_getf.get_event().wait()
		s.h_send.Start()
		s.h_send.Wait()


	def update_h(s):
		s.h_recv.Start()
		s.h_getf.get_event().wait()
		s.h_send.Start()

		s.h_recv.Wait()
		s.h_setf.set_fields(s.h_fhost_recv)
		s.h_send.Wait()




	neighbor_ranks = {}
	directions = ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']
	for direction in directions:
		target = neighbor_ranks[direction]
		if type(target) == int:
			e_func_list.append(
		else:
			pass




