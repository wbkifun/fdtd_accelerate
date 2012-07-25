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

        # pad_str_list
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

        # program
        dtype_str_list = { \
                'single':['float', 'xmmintrin.h', 'ps', '__m128', '4'], \
                'double':['double', 'emmintrin.h', 'pd', '__m128d', '2'] }[precision_float]

        macros = [ \
                'OMP_HEADER', 'OMP_FOR_E', 'OMP_FOR_H', \
                'ARGS_CE', 'INIT_CE', 'PRIVATE_CE', 'CEX', 'CEY', 'CEZ', \
                'ARGS_CH', 'INIT_CH', 'PRIVATE_CH', 'CHX', 'CHY', 'CHZ', \
                'DTYPE', 'MM_HEADER', 'PSD', 'TYPE128', 'INCRE', \
                'MASK_H', 'MASK_EXY', 'MASK']

        values = ['', '', '', \
                '', 'ce=SET1(0.5)', '', '', '', '', \
                '', 'ch=SET1(0.5)', '', '', '', '', \
                ] + dtype_str_list + pad_str_list

        if use_cpu_core != 1:
            values[0] = '#include <omp.h>'

            omp_str = '' if use_cpu_core == 0 else 'omp_set_num_threads(%d);\n\t' % use_cpu_core
            values[1] = omp_str + '#pragma omp parallel for private(idx, i, j, k, hx0, hy0, hz0, h1, h2, e PRIVATE_CE)'
            values[2] = omp_str + '#pragma omp parallel for private(idx, i, j, k, ex0, ey0, ez0, e1, e2, h PRIVATE_CH)'

        if ce_on:
            values[3:9] = [ \
                    ', DTYPE *cex, DTYPE *cey, DTYPE *cez', 'ce', ', ce', \
                    'ce = LOAD(cex+idx);', 'ce = LOAD(cey+idx);', 'ce = LOAD(cez+idx);']
        if ch_on:
            values[9:15] = [ \
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

        set_args = lambda args, idx0, nmax: args[:3] + [idx0, nmax] + args[5:]
        e_args_dict = { \
                '': e_args, \
                'pre': set_args(e_args, nyz_pitch, 3*nyz_pitch), \
                'post': set_args(e_args, 0, nyz_pitch) }

        h_args_dict = { \
                '': h_args, \
                'pre': set_args(h_args, 0, 2*nyz_pitch), \
                'post': set_args(h_args, 2*nyz_pitch, 3*nyz_pitch) }

        # global variables
        self.mainf = fields
        self.program = program
        self.e_args_dict = e_args_dict
        self.h_args_dict = h_args_dict

        # append to the update list
        self.priority_type = 'core'
        fields.append_instance(self)


    def update_e(self, part=''):
        self.mainf.enqueue( self.program.update_e, self.e_args_dict[part] )


    def update_h(self, part=''):
        self.mainf.enqueue( self.program.update_h, self.h_args_dict[part] )
