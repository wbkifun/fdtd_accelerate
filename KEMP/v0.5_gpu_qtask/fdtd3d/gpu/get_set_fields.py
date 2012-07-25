import numpy as np
import pyopencl as cl

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
                'DTYPE', 'PRAGMA_fp64']

        nmax_str, xid_str, yid_str, zid_str = common_gpu.macro_replace_list(pt0, pt1)
        values = [nmax_str, xid_str, yid_str, zid_str, \
                '__global DTYPE *source', \
                'target[sub_idx]', 'source[idx]', '='] + \
                dtype_str_list

        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'copy.cl').read(), macros, values)
        program = cl.Program(fields.context, ksrc).build()

        # allocation
        source_bufs = [fields.get_buf(str_f) for str_f in str_fs]
        shape = common.shape_two_points(pt0, pt1, len(str_fs))

        host_array = np.zeros(shape, dtype=fields.dtype)
        split_host_array = np.split(host_array, len(str_fs))
        split_host_array_dict = dict( zip(str_fs, split_host_array) ) 

        target_buf = cl.Buffer( \
                fields.context, \
                cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, \
                hostbuf=host_array)

        # global variables
        self.mainf = fields
        self.program = program
        self.source_bufs = source_bufs
        self.target_buf = target_buf
        self.host_array = host_array
        self.split_host_array_dict = split_host_array_dict

        nmax = int(nmax_str)
        remainder = nmax % fields.ls
        self.gs = nmax if remainder == 0 else nmax - remainder + fields.ls 


    def get_event(self):
        nx, ny, nz_pitch = self.mainf.ns_pitch

        for shift_idx, source_buf in enumerate(self.source_bufs):
            self.mainf.enqueue(self.program.copy, \
                    [self.mainf.queue, (self.gs,), (self.mainf.ls,), \
                    nx, ny, nz_pitch, np.int32(shift_idx), self.target_buf, source_buf])

        evt = self.mainf.enqueue(cl.enqueue_copy, \
                [self.mainf.queue, self.host_array, self.target_buf])

        return evt


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
                'DTYPE', 'PRAGMA_fp64']

        nmax_str, xid_str, yid_str, zid_str = common_gpu.macro_replace_list(pt0, pt1)
        if is_array:
            values = [nmax_str, xid_str, yid_str, zid_str, \
                    '__global DTYPE *source', \
                    'target[idx]', 'source[sub_idx]', overwrite_str] + \
                    dtype_str_list
        else:
            values = [nmax_str, xid_str, yid_str, zid_str, \
                    'DTYPE source', \
                    'target[idx]', 'source', overwrite_str] + \
                    dtype_str_list

        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'copy.cl').read(), macros, values)
        program = cl.Program(fields.context, ksrc).build()

		# allocation
        target_bufs = [fields.get_buf(str_f) for str_f in str_fs]
        shape = common.shape_two_points(pt0, pt1, len(str_fs))

        if is_array:
            tmp_array = np.zeros(shape, dtype=fields.dtype) 
            source_buf = cl.Buffer( \
                    fields.context, \
                    cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, \
                    hostbuf=tmp_array)

        # global variabels and functions
        self.mainf = fields
        self.program = program
        self.target_bufs = target_bufs
        self.shape = shape

        nmax = int(nmax_str)
        remainder = nmax % fields.ls
        self.gs = nmax if remainder == 0 else nmax - remainder + fields.ls 

        if is_array:
            self.source_buf = source_buf
            self.set_fields = self.set_fields_spatial_value
        else:
            self.set_fields = self.set_fields_single_value


    def set_fields_spatial_value(self, value, wait_for=[]):
        common.check_value('value.dtype', value.dtype, self.mainf.dtype)
        common.check_value('value.shape', value.shape, [self.shape])
        nx, ny, nz_pitch = self.mainf.ns_pitch

        self.mainf.enqueue(cl.enqueue_copy, \
                [self.mainf.queue, self.source_buf, value], \
                wait_for)

        for shift_idx, target_buf in enumerate(self.target_bufs):
            self.mainf.enqueue(self.program.copy, \
                    [self.mainf.queue, (self.gs,), (self.mainf.ls,), \
                    nx, ny, nz_pitch, np.int32(shift_idx), target_buf, self.source_buf])


    def set_fields_single_value(self, value, wait_for=[]):
        nx, ny, nz_pitch = self.mainf.ns_pitch

        self.mainf.enqueue(self.program.copy, \
                [self.mainf.queue, (self.gs,), (self.mainf.ls,), \
                nx, ny, nz_pitch, np.int32(0), self.target_bufs[0], self.mainf.dtype(value)], \
                wait_for)

        for target_buf in self.target_bufs[1:]:
            self.mainf.enqueue(self.program.copy, \
                    [self.mainf.queue, (self.gs,), (self.mainf.ls,), \
                    nx, ny, nz_pitch, np.int32(0), target_buf, self.mainf.dtype(value)])
