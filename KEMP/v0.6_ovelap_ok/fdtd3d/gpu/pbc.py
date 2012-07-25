import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu, common_exchange
from kemp.fdtd3d.gpu import Fields


class Pbc:
    def __init__(self, fields, axes):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_type('axes', axes, str)

        assert len( set(axes).intersection(set('xyz')) ) > 0, 'axes option is wrong: %s is given' % repr(axes)

        # local variables
        nx, ny, nz = fields.ns
        dtype_str_list = fields.dtype_str_list

        # program
        macros = ['NMAX', 'IDX0', 'IDX1', 'DTYPE', 'PRAGMA_fp64']

        program_dict = {}
        gs_dict = {}
        for axis in list(axes):
            program_dict[axis] = {}

            for e_or_h in ['e', 'h']:
                pt0 = common_exchange.pt0_dict(nx, ny, nz)[axis][e_or_h]['get']
                pt1 = common_exchange.pt1_dict(nx, ny, nz)[axis][e_or_h]['get']
                nmaxi_str, xid_str, yid_str, zid_str = common_gpu.macro_replace_list(pt0, pt1)
                idx0_str = '%s*ny*nz + %s*nz + %s' % (xid_str, yid_str, zid_str)

                pt0 = common_exchange.pt0_dict(nx, ny, nz)[axis][e_or_h]['set']
                pt1 = common_exchange.pt1_dict(nx, ny, nz)[axis][e_or_h]['set']
                nmax_str, xid_str, yid_str, zid_str = common_gpu.macro_replace_list(pt0, pt1)
                idx1_str = '%s*ny*nz + %s*nz + %s' % (xid_str, yid_str, zid_str)

                values = [nmax_str, idx0_str, idx1_str] + dtype_str_list

                ksrc = common.replace_template_code( \
                        open(common_gpu.src_path + 'copy_self.cl').read(), \
                        macros, values)
                program = cl.Program(fields.context, ksrc).build()
                program_dict[axis][e_or_h] = program

            nmax = int(nmax_str)
            remainder = nmax % fields.ls
            gs_dict[axis] = nmax if remainder == 0 else nmax - remainder + fields.ls 

        # global variables
        self.mainf = fields
        self.axes = axes
        self.program_dict = program_dict
        self.gs_dict = gs_dict

        # append to the update list
        self.priority_type = 'pbc'
        self.mainf.append_instance(self)


    def update(self, e_or_h):
        nx, ny, nz_pitch = self.mainf.ns_pitch

        for axis in list(self.axes):
            gs = self.gs_dict[axis]

            for str_f in common_exchange.str_fs_dict[axis][e_or_h]:
                self.program_dict[axis][e_or_h].copy_self( \
                        self.mainf.queue, (gs,), (self.mainf.ls,), \
                        nx, ny, nz_pitch, self.mainf.get_buf(str_f) )


    def update_e(self):
        self.update('e')


    def update_h(self):
        self.update('h')
