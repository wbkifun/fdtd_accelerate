import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu
from fields import Fields


class Core:
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
                open(common_gpu.src_path + 'core.cl').read(), macros, values)
        print ksrc
        program = cl.Program(context, ksrc).build()

        # arguments
        e_args = ns_pitch + eh_bufs
        h_args = ns_pitch + eh_bufs
        if ce_on:
            e_args += ce_bufs
        if ch_on:
            h_args += ch_bufs

        # global variables and functions
        self.mainf = fields
        self.program = program
        self.e_args = e_args
        self.h_args = h_args

        # append to the update list
        self.priority_type = 'core'
        self.mainf.append_instance(self)


    def update_e(self):
        self.program.update_e(self.mainf.queue, (self.mainf.gs,), (self.mainf.ls,), *self.e_args)


    def update_h(self):
        self.program.update_h(self.mainf.queue, (self.mainf.gs,), (self.mainf.ls,), *self.h_args)
