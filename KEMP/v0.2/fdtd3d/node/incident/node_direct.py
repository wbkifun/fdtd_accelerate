import numpy as np
import types

from kemp.fdtd3d.util import common
from kemp.fdtd3d.node import NodeFields
from kemp.fdtd3d import gpu, cpu


class NodeDirectIncident:
    def __init__(self, node_fields, str_f, pt0, pt1, tfunc, spatial_value=1., is_overwrite=False):
        """
        """
        
        common.check_type('node_fields', node_fields, NodeFields)
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
        anx = nodef.accum_nx_list

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], nodef.ns, pt0, pt1):
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

        # allocation
        dummied_shape = common.shape_two_points(pt0, pt1, is_dummy=True)

        incident_list = []
        reduced_slices = []
        self.gpu, self.cpu = gpu, cpu
        for i, mainf in enumerate(mainf_list):
            nx0 = anx[i]
            nx1 = anx[i+1]
            overlap = common.intersection_two_lines((nx0, nx1), (pt0[0], pt1[0]))

            if overlap != None:
                x0, y0, z0 = pt0
                x1, y1, z1 = pt1

                shift_pt0 = (overlap[0]-x0, y0-y0, z0-z0)
                shift_pt1 = (overlap[1]-x0, y1-y0, z1-z0)
                shift_slices = [slice(p0, p1+1) for p0, p1 in zip(shift_pt0, shift_pt1)]
                if is_array:
                    reshaped_value = spatial_value.reshape(dummied_shape)
                    dummied_array = reshaped_value[shift_slices]
                    overlap_shape = common.shape_two_points(shift_pt0, shift_pt1)
                    split_value = dummied_array.reshape(overlap_shape).copy()
                else:
                    split_value = spatial_value

                local_pt0 = (overlap[0]-nx0, y0, z0)
                local_pt1 = (overlap[1]-nx0, y1, z1)
                incident_list.append( \
                        getattr(self, mainf.device_type). \
                        DirectIncident(mainf, str_f, local_pt0, local_pt1, \
                        tfunc, split_value, is_overwrite) )

        # global variables
        self.incident_list = incident_list
