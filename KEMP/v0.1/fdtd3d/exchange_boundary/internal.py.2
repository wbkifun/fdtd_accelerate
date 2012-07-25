import numpy as np
from kemp.fdtd3d import gpu, cpu


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
	def __init__(s, fields_list, axis):
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

		s.e_update_list = s.get_update_list(s.e_setfs, s.e_getfs)
		s.h_update_list = s.get_update_list(s.h_setfs, s.h_getfs)


	def get_update_list(s, setfs, getfs):
		update_list = []
		for setf, getf in zip(setfs, getfs):
			if setf.emf.device_type == 'gpu' and getf.emf.device_type == 'cpu': 
				update_list.append( s.update_explicit_wait )
			else:
				update_list.append( s.update_implicit_wait )

		return update_list


	def update_explicit_wait(s, setf, getf):
		getf.get_event().wait()
		setf.set_fields(getf.get_fields())


	def update_implicit_wait(s, setf, getf):
		setf.set_fields(getf.get_fields(), [getf.get_event()])


	def update_e(s):
		for update, setf, getf in zip(s.e_update_list, s.e_setfs, s.e_getfs):
			update(setf, getf)


	def update_h(s):
		for update, setf, getf in zip(s.h_update_list, s.h_setfs, s.h_getfs):
			update(setf, getf)
