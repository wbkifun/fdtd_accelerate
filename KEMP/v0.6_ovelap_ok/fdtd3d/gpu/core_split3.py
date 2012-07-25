import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu
from fields import Fields


class CoreSplit3:
    def __init__(self, fields):
        """
        """

        common.check_type('fields', fields, Fields)

        # local variables
        context = fields.context

        ns_pitch = fields.ns_pitch
        pad = fields.pad

        precision_float = fields.precision_float
        dtype_str_list = fields.dtype_str_list

        ce_on = fields.ce_on
        ch_on = fields.ch_on

        eh_bufs = fields.eh_bufs
        if ce_on:
            ce_bufs = fields.ce_bufs
        if ch_on:
            ch_bufs = fields.ch_bufs

        ls = fields.ls

        # program
        str_pad = '' if pad==0 else '-%s' % pad
        coeff_constant = {'single': '0.5f', 'double': '0.5'}[precision_float]

        macros = ['ARGS_CE', 'CEX', 'CEY', 'CEZ', \
                'ARGS_CH', 'CHX', 'CHY', 'CHZ', \
                'DX', 'PAD', 'DTYPE', 'PRAGMA_fp64']

        values = ['', coeff_constant, coeff_constant, coeff_constant, \
                '', coeff_constant, coeff_constant, coeff_constant, \
                str(ls), str_pad] + dtype_str_list

        if ce_on:
            values[:4] = [ \
                    ', __global DTYPE *cex, __global DTYPE *cey, __global DTYPE *cez', \
                    'cex[idx]', 'cey[idx]', 'cez[idx]']

        if ch_on:
            values[4:8] = [ \
                    ', __global DTYPE *chx, __global DTYPE *chy, __global DTYPE *chz', \
                    'chx[idx]', 'chy[idx]', 'chz[idx]']

        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'core_split.cl').read(), macros, values)
        program = cl.Program(context, ksrc).build()

        # arguments
        e_args = ns_pitch + eh_bufs
        h_args = ns_pitch + eh_bufs
        if ce_on:
            e_args += ce_bufs
        if ch_on:
            h_args += ch_bufs

        nx, ny, nz_pitch = ns_pitch
        nyzp = ny * nz_pitch
        e_args_dict = { \
                '': [np.int32(0), np.int32(nx*nyzp)] + e_args, \
                'pre': [np.int32(nyzp), np.int32(2*nyzp)] + e_args, \
                'mid': [np.int32(2*nyzp), np.int32(nx*nyzp)] + e_args, \
                'post': [np.int32(0), np.int32(nyzp)] + e_args}

        h_args_dict = { \
                '': [np.int32(0), np.int32(nx*nyzp)] + h_args, \
                'pre': [np.int32((nx-2)*nyzp), np.int32((nx-1)*nyzp)] + h_args, \
                'mid': [np.int32(0), np.int32((nx-2)*nyzp)] + h_args, \
                'post': [np.int32((nx-1)*nyzp), np.int32(nx*nyzp)] + h_args}

        gs = lambda n: int(n) if (n % fields.ls) == 0 else int(n - (n % fields.ls) + fields.ls)
        gs_dict = { \
                '': gs(nx*nyzp), \
                'pre': gs(nyzp), \
                'mid': gs((nx-2)*nyzp), \
                'post': gs(nyzp)}

        # global variables and functions
        self.mainf = fields
        self.program = program
        self.e_args_dict = e_args_dict
        self.h_args_dict = h_args_dict
        self.gs_dict = gs_dict

        # append to the update list
        #self.priority_type = 'core'
        #self.mainf.append_instance(self)


    def update_e(self, part=''):
        self.program.update_e(self.mainf.queue, (self.gs_dict[part],), (self.mainf.ls,), *self.e_args_dict[part])


    def update_h(self, part=''):
        self.program.update_h(self.mainf.queue, (self.gs_dict[part],), (self.mainf.ls,), *self.h_args_dict[part])
