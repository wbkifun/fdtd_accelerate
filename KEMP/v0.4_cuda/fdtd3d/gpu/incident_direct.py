import numpy as np
import types

from kemp.fdtd3d.util import common
from fields import Fields
from get_set_fields import SetFields


class IncidentDirect:
    def __init__(self, fields, str_f, pt0, pt1, tfunc, spatial_value=1., is_overwrite=False):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_value('str_f', str_f, ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))
        common.check_type('pt0', pt0, (list, tuple), int)
        common.check_type('pt1', pt1, (list, tuple), int)
        common.check_type('tfunc', tfunc, types.FunctionType)
        common.check_type('spatial_value', spatial_value, \
                (np.ndarray, np.number, types.FloatType, types.IntType) )
        common.check_type('is_overwrite', is_overwrite, bool)

        # local variables
        e_or_h = str_f[0]
        dtype = fields.dtype
        is_array = True if isinstance(spatial_value, np.ndarray) else False

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], fields.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

        if is_array:
            shape = common.shape_two_points(pt0, pt1)
            assert shape == spatial_value.shape, \
                    'shape mismatch : %s, %s' % (shape, spatial_value.shape)
            assert dtype == spatial_value.dtype, \
                    'dtype mismatch : %s, %s' % (dtype, spatial_value.dtype)
        else:
            spatial_value = dtype(spatial_value)

        # create the SetFields instance
        setf = SetFields(fields, str_f, pt0, pt1, is_array, is_overwrite)

        # global variables
        self.mainf = fields
        self.tfunc = tfunc
        self.setf = setf
        self.spatial_value = spatial_value

        self.e_or_h = e_or_h
        self.tstep = 1

        # append to the update list
        self.priority_type = 'incident'
        fields.append_instance(self)


    def update(self):
        self.value = self.mainf.dtype(self.tfunc(self.tstep)) * self.spatial_value
        self.setf.set_fields(self.value)
        self.tstep += 1


    def update_e(self):
        if self.e_or_h == 'e':
            self.update()


    def update_h(self):
        if self.e_or_h == 'h':
            self.update()
