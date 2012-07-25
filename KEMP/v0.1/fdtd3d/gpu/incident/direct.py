import numpy as np

from kemp.fdtd3d.gpu import SetFields


class DirectSrc:
	def __init__(s, fields, str_f, pt0, pt1, tfunc, spatial_arr=1):
        """
        """

		self.mainf = fields
		self.tfunc = tfunc
		self.spatial_arr = spatial_arr

        if isinstance(spatial_arr, np.ndarray):
            self.setf = SetFields(self.emf, str_f, pt0, pt1, True)
        else:
            self.setf = SetFields(self.emf, str_f, pt0, pt1)


	def update(s, tstep):
        """
        """

		self.setf.set_fields(self.tfunc(tstep) * self.spatial_arr)
