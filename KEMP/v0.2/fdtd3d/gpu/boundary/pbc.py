import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu
from kemp.fdtd3d.gpu import Fields


class Pbc:
    def __init__(self, fields, axis):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_value('axis', axis, ['x', 'y', 'z'])

        # local variables
        nx, ny, nz = fields.ns
        dtype_str_list = fields.dtype_str_list

        # program
        replace = lambda lst, idx, val: lst[:idx] + [val] + lst[idx+1:]

        base0 = {'e': [0, 0, 0], 'h': [1, 1, 1]}
        base1 = {'e': [nx-2, ny-2, nz-2], 'h': [nx-1, ny-1, nz-1]}
        axis_id = {'x':0, 'y':1, 'z':2}[axis]
        nn = fields.ns[axis_id]

        value_dict = {'e': [], 'h': []}
        for eh in ['e', 'h']:
            for idx in {'e': [0, nn-1], 'h':[nn-1, 0]}[eh]:
                pt0 = replace(base0[eh], axis_id, idx)
                pt1 = replace(base1[eh], axis_id, idx)
                nmax, xid, yid, zid = \
                        common_gpu.macro_replace_list(pt0, pt1)

                value_dict[eh].append( \
                        '%s*ny*nz + %s*nz + %s' % (xid, yid, zid) )

        macros = ['NMAX', 'IDX0', 'IDX1', 'DTYPE', 'PRAGMA_fp64']
        values_e = [nmax] + value_dict['e'] + dtype_str_list
        values_h = [nmax] + value_dict['h'] + dtype_str_list

        ksrc_e = common.replace_template_code( \
                open(common_gpu.src_path + 'copy_self.cl').read(), \
                macros, values_e)
        ksrc_h = common.replace_template_code( \
                open(common_gpu.src_path + 'copy_self.cl').read(), \
                macros, values_h)
        program_e = cl.Program(fields.context, ksrc_e).build()
        program_h = cl.Program(fields.context, ksrc_h).build()

        # global variables
        self.mainf = fields
        self.program_e = program_e
        self.program_h = program_h

        self.strfs_e = {'x':['ey','ez'], 'y':['ex','ez'], 'z':['ex','ey']}[axis]
        self.strfs_h = {'x':['hy','hz'], 'y':['hx','hz'], 'z':['hx','hy']}[axis]

        # append to the update list
        self.priority_type = 'pbc'
        self.mainf.append_instance(self)


    def update_e(self):
        nx, ny, nz_pitch = self.mainf.ns_pitch

        for strf in self.strfs_e:
            self.program_e.copy( \
                    self.mainf.queue, (self.mainf.gs,), (self.mainf.ls,), \
                    nx, ny, nz_pitch, self.mainf.get_buf(strf) )

    def update_h(self):
        nx, ny, nz_pitch = self.mainf.ns_pitch

        for strf in self.strfs_h:
            self.program_h.copy( \
                    self.mainf.queue, (self.mainf.gs,), (self.mainf.ls,), \
                    nx, ny, nz_pitch, self.mainf.get_buf(strf) )
