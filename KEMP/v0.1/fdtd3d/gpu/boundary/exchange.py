import numpy as np
from kemp.fdtd3d import common
from kemp.fdtd3d.gpu import GetFields, SetFields


class ExchangeFields:
	def __init__(s, fields_list, axis):
		emf_list = fields_list

		e_strfs, h_strfs, pt0, pt1 = common.get_strfs_pts_for_boundary(axis, *emf_list[0].ns)

		s.e_getfs, s.h_getfs = [], []
		s.e_setfs, s.h_setfs = [], []
		for emf in emf_list[1:]:
			s.e_getfs.append( GetFields(emf, e_strfs, pt0['-'], pt1['-']) )
			s.h_setfs.append( SetFields(emf, h_strfs, pt0['-'], pt1['-'], np.ndarray) )

		for emf in emf_list[:-1]:
			s.h_getfs.append( GetFields(emf, h_strfs, pt0['+'], pt1['+']) )
			s.e_setfs.append( SetFields(emf, e_strfs, pt0['+'], pt1['+'], np.ndarray) )


	def update_e(s):
		for getf, setf in zip(s.e_getfs, s.e_setfs):
			setf.set_fields(getf.get_fields(), [getf.get_event()])


	def update_h(s):
		for getf, setf in zip(s.h_getfs, s.h_setfs):
			setf.set_fields(getf.get_fields(), [getf.get_event()])
