import numpy as np

from kemp.fdtd3d.util import common
from kemp.fdtd3d.cpu import Fields


class Pbc:
    def __init__(self, fields, axis):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_value('axis', axis, ['x', 'y', 'z'])

        mtype = fields.mpi_type
        if axis == 'x' and mtype in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
            raise ValueError, 'The fields.mpi_type is \'%s\'. The buffer instance is only permit the pbc operation along y and z axes' % mtype

        # local variables
        nx, ny, nz = fields.ns
        axis_id = {'x':0, 'y':1, 'z':2}[axis]

        # slice indices
        replace = lambda lst, idx, val: lst[:idx] + [val] + lst[idx+1:]

        slices_e = [slice(None, -1), slice(None, -1), slice(None, nz-1)]
        slices_h = [slice(1, None), slice(1, None), slice(1, nz)]

        slices_e_src = replace(slices_e, axis_id, slice(None, 1))
        slices_h_dest = replace(slices_h, axis_id, slice(None, 1))
        if axis == 'z':
            slices_e_dest = replace(slices_e, axis_id, slice(nz-1, nz))
            slices_h_src = replace(slices_h, axis_id, slice(nz-1, nz))
        else:
            slices_e_dest = replace(slices_e, axis_id, slice(-1, None))
            slices_h_src = replace(slices_h, axis_id, slice(-1, None))

        # global variables
        self.mainf = fields
        self.slices_dict = { \
                'e_src': fields.split_slices_dict('e', slices_e_src), \
                'e_dest': fields.split_slices_dict('e', slices_e_dest), \
                'h_src': fields.split_slices_dict('h', slices_h_src), \
                'h_dest': fields.split_slices_dict('h', slices_h_dest) }

        self.strfs = {\
                'x': {'e': ['ey','ez'], 'h': ['hy','hz']}, \
                'y': {'e': ['ex','ez'], 'h': ['hx','hz']}, \
                'z': {'e': ['ex','ey'], 'h': ['hx','hy']} }[axis]

        # append to the update list
        self.priority_type = 'pbc'
        self.mainf.append_instance(self)


    def update(self, e_or_h, part):
        sl_src = self.slices_dict['%s_src' % e_or_h][part]
        sl_dest = self.slices_dict['%s_dest' % e_or_h][part]

        for strf in self.strfs[e_or_h]:
            f = self.mainf.get(strf)
            f[sl_dest] = f[sl_src]


    def update_e(self, part=''):
        self.mainf.enqueue(self.update, ['e', part])


    def update_h(self, part=''):
        self.mainf.enqueue(self.update, ['h', part])
