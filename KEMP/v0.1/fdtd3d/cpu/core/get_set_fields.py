import numpy as np
from kemp.fdtd3d import common


class GetFields:
	def __init__(s, fields, str_fs, pt0, pt1):
		s.emf = fields
		if type(str_fs) == str:
			s.str_fs = [str_fs,]
		else:
			s.str_fs = str_fs

		s.str_fs_size = len(s.str_fs)
		s.slidx = common.get_slice_index(pt0, pt1)
		s.shape = common.get_shape(pt0, pt1)
		s.shape[0] *= s.str_fs_size
		s.fhost = np.zeros(s.shape, dtype=s.emf.dtype)
		s.split_fs = dict( zip(s.str_fs, np.array_split(s.fhost, s.str_fs_size)) ) 


	def copy_to_fhost(s):
		for str_f in s.str_fs:
			s.split_fs[str_f][:] = s.emf[str_f][s.slidx]


	def get_event(s):
		s.emf.enqueue(s.copy_to_fhost, lock=True)
		return s.emf.qtask


	def get_fields(s, str_f=None):
		if str_f == None:
			return s.fhost
		else:
			return s.split_fs[str_f]



class SetFields:
	def __init__(s, fields, str_fs, pt0, pt1, dtype_values=None):
		s.emf = fields
		if type(str_fs) == str:
			s.str_fs = [str_fs,]
		else:
			s.str_fs = str_fs

		s.str_fs_size = len(s.str_fs)
		s.slidx = common.get_slice_index(pt0, pt1)

		if dtype_values == np.ndarray:
			s.func = s.set_fields_spatial_values
		else:
			s.func = s.set_fields_single_value


	def set_fields_spatial_values(s, values):
		for str_f, ndarr in zip(s.str_fs, np.array_split(values, s.str_fs_size)):
			s.emf.__dict__[str_f][s.slidx] = ndarr


	def set_fields_single_value(s, value):
		for str_f in s.str_fs:
			s.emf.__dict__[str_f][s.slidx] = value


	def set_fields(s, values, wait_list=[]):
		s.emf.enqueue(s.func, [values], wait_list)
