import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu
from kemp.fdtd3d.gpu import GetFields, SetFields


class PbcInt:
	def __init__(s, fields, axis):
		s.emf = fields

		s.e_strfs = {'x':['ey','ez'], 'y':['ex','ez'], 'z':['ex','ey']}[axis]
		s.h_strfs = {'x':['hy','hz'], 'y':['hx','hz'], 'z':['hx','hy']}[axis]

		macros = ['NMAX', 'IDX1', 'IDX2']
		e_vals = {
				'x': ['ny*nz', '(nx-1)*ny*nz + gid', 'gid'],
				'y': ['nx*nz', '(gid/nz)*ny*nz + (ny-1)*nz + gid', '(gid/nz)*ny*nz + gid'],
				'z': ['nx*ny', '(gid/ny)*ny*nz + gid*nz + (nz-1)', '(gid/ny)*ny*nz + gid*nz'] }[axis]
		h_vals = {
				'x': ['ny*nz', 'gid', '(nx-1)*ny*nz + gid'],
				'y': ['nx*nz', '(gid/nz)*ny*nz + gid', '(gid/nz)*ny*nz + (ny-1)*nz + gid'],
				'z': ['nx*ny', '(gid/ny)*ny*nz + gid*nz', '(gid/ny)*ny*nz + gid*nz + (nz-1)'] }[axis]

		e_ksrc = common.replace_template_code(open(common_gpu.src_path + '/copy.cl').read(), macros, e_vals)
		h_ksrc = common.replace_template_code(open(common_gpu.src_path + '/copy.cl').read(), macros, h_vals)
		s.program_e = cl.Program(s.emf.context, e_ksrc).build()
		s.program_h = cl.Program(s.emf.context, h_ksrc).build()


	def update_e(s):
		for strf in s.e_strfs:
			s.program_e.copy(s.emf.queue, (s.emf.gs,), (s.emf.ls,), *(s.emf.ns + [s.emf.get_buffer(strf)]))


	def update_h(s):
		for strf in s.h_strfs:
			s.program_h.copy(s.emf.queue, (s.emf.gs,), (s.emf.ls,), *(s.emf.ns + [s.emf.get_buffer(strf)]))



class PbcExt:
	def __init__(s, fields_list, axis):
		emf_list = fields_list

		e_strfs, h_strfs, pt0, pt1 = common_gpu.get_strfs_pts(axis, *emf_list[0].ns)

		s.e_getf = GetFields(emf_list[0], e_strfs, pt0['-'], pt1['-'])
		s.h_setf = SetFields(emf_list[0], h_strfs, pt0['-'], pt1['-'], np.ndarray, True)

		s.h_getf = GetFields(emf_list[-1], h_strfs, pt0['+'], pt1['+'])
		s.e_setf = SetFields(emf_list[-1], e_strfs, pt0['+'], pt1['+'], np.ndarray, True)


	def update_e(s):
		s.e_setf.set_fields(s.e_getf.get_fields(), s.e_getf.get_event())


	def update_h(s):
		s.h_setf.set_fields(s.h_getf.get_fields(), s.h_getf.get_event())
