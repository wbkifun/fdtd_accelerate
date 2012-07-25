import numpy as np
from ctypes import c_int

from kemp.fdtd3d.util import common, common_cpu
from fields import Fields


class Core:
    def __init__(self, fields):
        """
        """

        common.check_type('fields', fields, Fields)

        # local variables
        precision_float = fields.precision_float
        use_cpu_core = fields.use_cpu_core
        dtype = fields.dtype

        nx, ny, nz_pitch = ns_pitch = fields.ns_pitch
        align_size = fields.align_size
        pad = fields.pad

        ce_on = fields.ce_on
        ch_on = fields.ch_on

        ehs = fields.ehs
        if ce_on:
            ces = fields.ces
        if ch_on:
            chs = fields.chs

        # program
        dtype_str_list = { \
                'single':['float', 'xmmintrin.h', 'ps', '__m128', '4'], \
                'double':['double', 'emmintrin.h', 'pd', '__m128d', '2'] }[precision_float]
        pad_str_list = []
        pad_str_append = lambda mask: pad_str_list.append( str(list(mask)).strip('[]') )
        mask0 = np.ones(align_size, 'i')

        mask_h = mask0.copy()
        mask_h[0] = 0
        pad_str_append(mask_h)

        mask_exy = mask0.copy()
        mask_exy[-(pad+1):] = 0
        pad_str_append(mask_exy)

        mask = mask0.copy()
        if pad != 0:
            mask[-pad:] = 0
        pad_str_append(mask)

        macros = [ \
                'ARGS_CE', 'INIT_CE', 'PRIVATE_CE', 'CEX', 'CEY', 'CEZ', \
                'ARGS_CH', 'INIT_CH', 'PRIVATE_CH', 'CHX', 'CHY', 'CHZ', \
                'OMP_SET_NUM_THREADS', \
                'DTYPE', 'MM_HEADER', 'PSD', 'TYPE128', 'INCRE', \
                'MASK_H', 'MASK_EXY', 'MASK']

        values = [ \
                '', 'ce=SET1(0.5)', '', '', '', '', \
                '', 'ch=SET1(0.5)', '', '', '', '', \
                ''] + dtype_str_list + pad_str_list

        if use_cpu_core != 0:
            values[12] = 'omp_set_num_threads(%d);' % use_cpu_core

        if ce_on:
            values[:6] = [ \
                    ', DTYPE *cex, DTYPE *cey, DTYPE *cez', 'ce', ', ce', \
                    'ce = LOAD(cex+idx);', 'ce = LOAD(cey+idx);', 'ce = LOAD(cez+idx);']
        if ch_on:
            values[6:12] = [ \
                    ', DTYPE *chx, DTYPE *chy, DTYPE *chz', 'ch', ', ch', \
                    'ch = LOAD(chx+idx);', 'ch = LOAD(chy+idx);', 'ch = LOAD(chz+idx);']

        ksrc = common.replace_template_code( \
                open(common_cpu.src_path + 'core.c').read(), macros, values)
        program = common_cpu.build_clib(ksrc)

        carg = np.ctypeslib.ndpointer(dtype, ndim=3, \
                shape=tuple(ns_pitch), flags='C_CONTIGUOUS, ALIGNED')
        argtypes = [c_int, c_int, c_int, c_int, c_int] + \
                [carg for i in xrange(6)]
        program.update_e.argtypes = argtypes
        program.update_e.restype = None
        program.update_h.argtypes = argtypes
        program.update_h.restype = None

        # arguments
        nyz_pitch = ny * nz_pitch
        e_args = ns_pitch + [0, nx*nyz_pitch] + ehs
        h_args = ns_pitch + [0, nx*nyz_pitch] + ehs
        if ce_on:
            program.update_e.argtypes += [carg for i in xrange(3)]
            e_args += ces
        if ch_on:
            program.update_h.argtypes += [carg for i in xrange(3)]
            h_args += chs

        pre_e_args = e_args[:]
        pre_e_args[3:5] = [(nx-2)*nyz_pitch, nx*nyz_pitch]
        mid_e_args = e_args[:]
        mid_e_args[3:5] = [nyz_pitch, (nx-2)*nyz_pitch]
        post_e_args = e_args[:]
        post_e_args[3:5] = [0, nyz_pitch]

        pre_h_args = h_args[:]
        pre_h_args[3:5] = [0, 2*nyz_pitch]
        mid_h_args = h_args[:]
        mid_h_args[3:5] = [2*nyz_pitch, (nx-1)*nyz_pitch]
        post_h_args = h_args[:]
        post_h_args[3:5] = [(nx-1)*nyz_pitch, nx*nyz_pitch]

        # global variables
        self.mainf = fields
        self.e_args = e_args
        self.h_args = h_args
        self.program = program

        self.e_args_dict = {'':e_args, \
                'pre':pre_e_args, 'mid':mid_e_args, 'post':post_e_args}
        self.h_args_dict = {'':h_args, \
                'pre':pre_h_args, 'mid':mid_h_args, 'post':post_h_args}

        # append to the update list
        self.priority_type = 'core'
        fields.append_instance(self)


    def update_e(self, part=''):
        self.mainf.enqueue( self.program.update_e, self.e_args_dict[part] )


    def update_h(self, part=''):
        self.mainf.enqueue( self.program.update_h, self.h_args_dict[part] )
