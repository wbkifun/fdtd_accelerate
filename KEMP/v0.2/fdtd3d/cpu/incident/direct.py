import numpy as np
import types

from kemp.fdtd3d.util import common
from kemp.fdtd3d.cpu import Fields, SetFields


class DirectIncident:
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
        func_dict = {}

        pts_dict = fields.split_points_dict(e_or_h, pt0, pt1)
        for part, pts in pts_dict.items():
            if pts == None:
                func_dict[part] = lambda a='': None
            else:
                func_dict[part] = SetFields(fields, str_f, \
                    pts[0], pts[1], is_array, is_overwrite).set_fields

        if is_array:
            spatial_array_dict = {}

            for part, pts in pts_dict.items():
                if pts == None:
                    spatial_array_dict[part] = 0

                else:
                    slices0 = [slice(p0, p1+1) for p0, p1 in zip(pt0, pt1)]
                    slices1 = [slice(p0, p1+1) for p0, p1 in zip(pts[0], pts[1])]
                    overlap_slices = common.intersection_two_slices(fields.ns, slices0, slices1)

                    shift_slices = []
                    for sl, p0 in zip(overlap_slices, pt0):
                        s0, s1 = sl.start, sl.stop
                        shift_slices.append( slice(s0-p0, s1-p0) )

                    dummied_shape = common.shape_two_points(pt0, pt1, is_dummy=True)
                    reshaped_value = spatial_value.reshape(dummied_shape)
                    dummied_array = reshaped_value[shift_slices]

                    overlap_shape = common.shape_two_points(pts[0], pts[1])
                    spatial_array_dict[part] = dummied_array.reshape(overlap_shape)
                    
        # global variables and functions
        self.mainf = fields
        self.dtype = dtype
        self.tfunc = tfunc
        self.func_dict = func_dict
        self.e_or_h = e_or_h
        self.tstep = 1
        
        if is_array:
            self.spatial_array_dict = spatial_array_dict
            self.update = self.update_spatial_value
        else:
            self.spatial_value = spatial_value
            self.update = self.update_single_value

        # append to the update list
        self.priority_type = 'incident'
        fields.append_instance(self)


    def update_spatial_value(self, part):
        value = self.dtype(self.tfunc(self.tstep)) * self.spatial_array_dict[part]
        self.func_dict[part](value)

        if part in ['', 'post']:
            self.tstep += 1


    def update_single_value(self, part):
        value = self.dtype(self.tfunc(self.tstep)) * self.spatial_value
        self.func_dict[part](value)

        if part in ['', 'post']:
            self.tstep += 1


    def update_e(self, part=''):
        if self.e_or_h == 'e':
            self.mainf.enqueue(self.update, [part])


    def update_h(self, part=''):
        if self.e_or_h == 'h':
            self.mainf.enqueue(self.update, [part])
