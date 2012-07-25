import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from kemp.fdtd3d.util import common, common_gpu
from fields import Fields


class Core:
    def __init__(self, fields):
        """
        """

        common.check_type('fields', fields, Fields)

        # local variables
        ns_pitch = fields.ns_pitch
        pad = fields.pad

        precision_float = fields.precision_float
        dtype_str_list = fields.dtype_str_list

        ce_on = fields.ce_on
        ch_on = fields.ch_on

        eh_bufs = fields.eh_bufs
        ce_bufs = fields.ce_bufs
        ch_bufs = fields.ch_bufs

        bs = fields.bs

        # program
        str_pad = '' if pad==0 else '-%s' % pad
        coeff_constant = {'single': '0.5f', 'double': '0.5'}[precision_float]

        macros = ['ARGS_CE', 'CEX', 'CEY', 'CEZ', \
                'ARGS_CH', 'CHX', 'CHY', 'CHZ', \
                'DX', 'PAD', 'DTYPE']

        values = ['', coeff_constant, coeff_constant, coeff_constant, \
                '', coeff_constant, coeff_constant, coeff_constant, \
                str(bs[0]), str_pad] + dtype_str_list

        if ce_on:
            values[:4] = [ \
                    ', DTYPE *cex, DTYPE *cey, DTYPE *cez', \
                    'cex[idx]', 'cey[idx]', 'cez[idx]']

        if ch_on:
            values[4:8] = [ \
                    ', DTYPE *chx, DTYPE *chy, DTYPE *chz', \
                    'chx[idx]', 'chy[idx]', 'chz[idx]']

        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'core.cu').read(), macros, values)
        program = SourceModule(ksrc)
        kernel_update_e = program.get_function('update_e')
        kernel_update_h = program.get_function('update_h')

        # arguments
        args = ns_pitch + eh_bufs
        e_args = args + ce_bufs if ce_on else args
        h_args = args + ch_bufs if ch_on else args

        kernel_update_e.prepare([type(arg) for arg in e_args])
        kernel_update_h.prepare([type(arg) for arg in h_args])

        # global variables and functions
        self.mainf = fields
        self.e_args = e_args
        self.h_args = h_args
        self.kernel_update_e = kernel_update_e
        self.kernel_update_h = kernel_update_h

        # append to the update list
        self.priority_type = 'core'
        self.mainf.append_instance(self)


    def update_e(self):
        #self.kernel_update_e.prepared_call(self.mainf.gs, self.mainf.bs, *self.e_args)
        #self.kernel_update_e.prepared_call((8192,1), (256,1,1), *self.e_args)
        self.kernel_update_e.prepared_call((61440,1), (256,1,1), *self.e_args)


    def update_h(self):
        #self.kernel_update_h.prepared_async_call(self.mainf.gs, self.mainf.bs, self.mainf.stream, *self.h_args)
        #self.kernel_update_h.prepared_call((8192,1), (256,1,1), *self.h_args)
        self.kernel_update_h.prepared_call((61440,1), (256,1,1), *self.h_args)
