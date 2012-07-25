from kemp.fdtd3d.cpu import SetFields


class DirectSrc:
	def __init__(s, fields, str_f, pt0, pt1, tfunc, spatial_arr=1.):
		s.emf = fields
		s.tfunc = tfunc
		s.spatial_arr = spatial_arr

		s.setf = SetFields(s.emf, str_f, pt0, pt1, type(spatial_arr))


	def update(s, tstep):
		s.setf.set_fields(s.tfunc(tstep) * s.spatial_arr)
