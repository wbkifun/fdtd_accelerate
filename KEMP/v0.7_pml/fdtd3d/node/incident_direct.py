import numpy as np
import types

from kemp.fdtd3d.util import common
from kemp.fdtd3d.node import Fields
from kemp.fdtd3d import cpu


class IncidentDirect:
    def __init__(self, node_fields, str_f, pt0, pt1, tfunc, spatial_value=1., is_overwrite=False):
        """
        """
        
        common.check_type('node_fields', node_fields, Fields)
        common.check_value('str_f', str_f, ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))
        common.check_type('pt0', pt0, (list, tuple), int)
        common.check_type('pt1', pt1, (list, tuple), int)
        common.check_type('tfunc', tfunc, types.FunctionType)
        common.check_type('spatial_value', spatial_value, \
                (np.ndarray, np.number, types.FloatType, types.IntType) )
        common.check_type('is_overwrite', is_overwrite, bool)

        # local variables
        nodef = node_fields
        dtype = nodef.dtype
        is_array = True if isinstance(spatial_value, np.ndarray) else False
        mainf_list = nodef.mainf_list
        buffer_dict = nodef.buffer_dict
        anx = nodef.accum_nx_list
        nx, ny, nz = nodef.ns

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], nodef.ns, pt0, pt1):
            start, end = 0, n
            if buffer_dict.has_key(axis+'+'):
                end = n + 1
            if buffer_dict.has_key(axis+'-'):
                start = -1
            common.check_value('pt0 %s' % axis, p0, range(start, end))
            common.check_value('pt1 %s' % axis, p1, range(start, end))

        if is_array:
            shape = common.shape_two_points(pt0, pt1)
            assert shape == spatial_value.shape, \
                    'shape mismatch : %s, %s' % (shape, spatial_value.shape)
            assert dtype == spatial_value.dtype, \
                    'dtype mismatch : %s, %s' % (dtype, spatial_value.dtype)
        else:
            spatial_value = dtype(spatial_value)

        # global valriables
        self.str_f = str_f
        self.pt0 = pt0
        self.pt1 = pt1
        self.tfunc= tfunc
        self.spatial_value = spatial_value
        self.is_overwrite = is_overwrite

        self.is_array = is_array
        self.cpu = cpu
        if 'gpu' in [f.device_type for f in nodef.updatef_list]:
            from kemp.fdtd3d import gpu
            self.gpu = gpu

        # create IncidentDirect instance
        for i, mainf in enumerate(mainf_list):
            fields_pt0 = (anx[i], 0, 0)
            fields_pt1 = (anx[i+1]-1, ny-1, nz-1)
            overlap = common.overlap_two_regions(fields_pt0, fields_pt1, pt0, pt1)

            if overlap != None:
                self.create_instance(mainf, fields_pt0, fields_pt1, overlap[0], overlap[1])


        # for buffer
        for direction, buffer in buffer_dict.items():
            fields_pt0 = { \
                    'x+': (anx[-1]-1, 0, 0), \
                    'y+': (0, ny-2, 0), \
                    'z+': (0, 0, nz-2), \
                    'x-': (-1, 0, 0), \
                    'y-': (0, -1, 0), \
                    'z-': (0, 0, -1) }[direction]

            fields_pt1 = { \
                    'x+': (anx[-1]+1, ny-1, nz-1), \
                    'y+': (nx-1, ny, nz-1), \
                    'z+': (nx-1, ny-1, nz), \
                    'x-': (1, ny-1, nz-1), \
                    'y-': (nx-1, 1, nz-1), \
                    'z-': (nx-1, ny-1, 1) }[direction]

            overlap = common.overlap_two_regions(fields_pt0, fields_pt1, pt0, pt1)
            if overlap != None:
                self.create_instance(buffer, fields_pt0, fields_pt1, overlap[0], overlap[1])


    def create_instance(self, fields, fields_pt0, fields_pt1, overlap_pt0, overlap_pt1):
        sx0, sy0, sz0 = fields_pt0
        ox0, oy0, oz0 = overlap_pt0
        ox1, oy1, oz1 = overlap_pt1

        subf_pt0 = (ox0-sx0, oy0-sy0, oz0-sz0)
        subf_pt1 = (ox1-sx0, oy1-sy0, oz1-sz0)

        if self.is_array:
            x0, y0, z0 = self.pt0
            svalue_pt0 = (ox0-x0, oy0-y0, oz0-z0)
            svalue_pt1 = (ox1-x0, oy1-y0, oz1-z0)

            dummied_shape = common.shape_two_points(self.pt0, self.pt1, is_dummy=True)
            dummied_shaped_svalue = self.spatial_value.reshape(dummied_shape)
            dummied_slices = [slice(p0, p1+1) for p0, p1 in zip(svalue_pt0, svalue_pt1)]
            subdomain_shape = common.shape_two_points(svalue_pt0, svalue_pt1)
            svalue_subdomain = dummied_shaped_svalue[dummied_slices].reshape(subdomain_shape).copy()
        else:
            svalue_subdomain = self.spatial_value

        getattr(self, fields.device_type).IncidentDirect( \
                fields, self.str_f, subf_pt0, subf_pt1, self.tfunc, svalue_subdomain, self.is_overwrite)
