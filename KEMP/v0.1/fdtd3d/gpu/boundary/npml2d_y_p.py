import numpy as np
import pyopencl as cl
from kemp.fdtd3d import common, common_gpu


class Npml:
	def __init__(s, fields, ca0, cb0, ca1, cb1, ca2, cb2):
		s.emf = fields

		# allocations
		nx, ny, nz = s.emf.ns
		pe = np.zeros((nx, nz), dtype=s.emf.dtype)
		ph = np.zeros((2, nx, nz), dtype=s.emf.dtype)

		mf = cl.mem_flags
		s.pez, s.pex = [cl.Buffer(s.emf.context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=pe) for i in range(2)]
		s.phz, s.phx = [cl.Buffer(s.emf.context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=ph) for i in range(2)]

		# program
		ksrc = open(common_gpu.src_path + 'npml2d_y_p.cl').read()
		s.program = cl.Program(s.emf.context, ksrc).build()
		s.args_e = s.emf.ns + \
				[s.emf.ez, s.emf.ex, s.emf.hz, s.emf.hx, 
				s.pez, s.pex, s.phz, s.phx, 
				np.float32(cb0), 
				np.float32(ca1), np.float32(cb1), 
				np.float32(ca2), np.float32(cb2)]
		s.args_h = s.emf.ns + \
				[s.emf.ez, s.emf.ex, s.emf.hz, s.emf.hx, 
				s.pez, s.pex, s.phz, s.phx, 
				np.float32(ca0), np.float32(cb0), 
				np.float32(cb1), 
				np.float32(cb2)]


	def update_e(s):
		s.program.update_e(s.emf.queue, (s.emf.gs,), (s.emf.ls,), *s.args_e)


	def update_h(s):
		s.program.update_h(s.emf.queue, (s.emf.gs,), (s.emf.ls,), *s.args_h)
