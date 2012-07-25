import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from kemp.fdtd3d.util import common, common_gpu
from fields import Fields


class GetFields:
    def __init__(self, fields, str_f, pt0, pt1):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_type('str_f', str_f, (str, list, tuple), str)
        common.check_type('pt0', pt0, (list, tuple), (int, float))
        common.check_type('pt1', pt1, (list, tuple), (int, float))

        pt0 = list( common.convert_indices(fields.ns, pt0) )
        pt1 = list( common.convert_indices(fields.ns, pt1) )

        # local variables
        str_fs = common.convert_to_tuple(str_f)
        dtype_str_list = fields.dtype_str_list

        for strf in str_fs:
            strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
            common.check_value('str_f', strf, strf_list)

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], fields.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

        # program
        macros = ['NMAX', 'XID', 'YID', 'ZID', \
                'ARGS', \
                'TARGET', 'SOURCE', 'OVERWRITE', \
                'DTYPE']

        values = common_gpu.macro_replace_list(pt0, pt1) + \
                ['DTYPE *source', \
                'target[sub_idx]', 'source[idx]', '='] + \
                dtype_str_list

        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'copy.cu').read(), macros, values)
        program = SourceModule(ksrc)
        kernel_copy = program.get_function('copy')

        # allocation
        source_bufs = [fields.get_buf(str_f) for str_f in str_fs]
        shape = common.shape_two_points(pt0, pt1, len(str_fs))

        host_array = np.zeros(shape, fields.dtype)
        split_host_array = np.split(host_array, len(str_fs))
        split_host_array_dict = dict( zip(str_fs, split_host_array) ) 
        target_buf = cuda.to_device(host_array)

        # global variables
        self.mainf = fields
        self.kernel_copy = kernel_copy
        self.source_bufs = source_bufs
        self.target_buf = target_buf
        self.host_array = host_array
        self.split_host_array_dict = split_host_array_dict


    def wait(self):
        nx, ny, nz_pitch = self.mainf.ns_pitch

        for shift_idx, source_buf in enumerate(self.source_bufs):
            self.kernel_copy( \
                    nx, ny, nz_pitch, np.int32(shift_idx), self.target_buf, source_buf, \
                    grid=self.mainf.gs, block=self.mainf.bs)

        cuda.memcpy_dtoh(self.host_array, self.target_buf)


    def get_fields(self, str_f=''):
        if str_f == '':
            return self.host_array
        else:
            return self.split_host_array_dict[str_f]



class SetFields:
    def __init__(self, fields, str_f, pt0, pt1, is_array=False, is_overwrite=True):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_type('str_f', str_f, (str, list, tuple), str)
        common.check_type('pt0', pt0, (list, tuple), (int, float))
        common.check_type('pt1', pt1, (list, tuple), (int, float))
        common.check_type('is_array', is_array, bool)
        common.check_type('is_overwrite', is_overwrite, bool)

        pt0 = list( common.convert_indices(fields.ns, pt0) )
        pt1 = list( common.convert_indices(fields.ns, pt1) )

        # local variables
        str_fs = common.convert_to_tuple(str_f)
        dtype_str_list = fields.dtype_str_list
        overwrite_str = {True: '=', False: '+='}[is_overwrite]

        for strf in str_fs:
            strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
            common.check_value('str_f', strf, strf_list)

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], fields.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

		# program
        macros = ['NMAX', 'XID', 'YID', 'ZID', \
                'ARGS', \
                'TARGET', 'SOURCE', 'OVERWRITE', \
                'DTYPE']

        if is_array:
            values = common_gpu.macro_replace_list(pt0, pt1) + \
                    ['DTYPE *source', \
                    'target[idx]', 'source[sub_idx]', overwrite_str] + \
                    dtype_str_list
        else:
            values = common_gpu.macro_replace_list(pt0, pt1) + \
                    ['DTYPE source', \
                    'target[idx]', 'source', overwrite_str] + \
                    dtype_str_list

        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'copy.cu').read(), macros, values)
        program = SourceModule(ksrc)
        kernel_copy = program.get_function('copy')

		# allocation
        target_bufs = [fields.get_buf(str_f) for str_f in str_fs]
        shape = common.shape_two_points(pt0, pt1, len(str_fs))

        if is_array:
            tmp_array = np.zeros(shape, fields.dtype)
            source_buf = cuda.to_device(tmp_array)

        # global variabels and functions
        self.mainf = fields
        self.kernel_copy = kernel_copy
        self.target_bufs = target_bufs
        self.shape = shape

        if is_array:
            self.source_buf = source_buf
            self.set_fields = self.set_fields_spatial_value
        else:
            self.set_fields = self.set_fields_single_value


    def set_fields_spatial_value(self, value):
        common.check_value('value.dtype', value.dtype, self.mainf.dtype)
        common.check_value('value.shape', value.shape, [self.shape])
        nx, ny, nz_pitch = self.mainf.ns_pitch

        cuda.memcpy_htod(self.source_buf, value)

        for shift_idx, target_buf in enumerate(self.target_bufs):
            self.kernel_copy( \
                    nx, ny, nz_pitch, np.int32(shift_idx), target_buf, self.source_buf, \
                    grid=self.mainf.gs, block=self.mainf.bs)


    def set_fields_single_value(self, value):
        nx, ny, nz_pitch = self.mainf.ns_pitch

        for target_buf in self.target_bufs:
            self.kernel_copy( \
                    nx, ny, nz_pitch, np.int32(0), target_buf, self.mainf.dtype(value), \
                    grid=self.mainf.gs, block=self.mainf.bs)
