import numpy as np

from kemp.fdtd3d.util import common
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

        for strf in str_fs:
            strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
            common.check_value('str_f', strf, strf_list)

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], fields.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

        # allocation
        shape = common.shape_two_points(pt0, pt1, len(str_fs))
        host_array = np.zeros(shape, dtype=fields.dtype)

        split_host_array = np.split(host_array, len(str_fs))
        split_host_array_dict = dict( zip(str_fs, split_host_array) ) 

        # global variables
        self.mainf = fields
        self.str_fs = str_fs
        self.slice_xyz = common.slices_two_points(pt0, pt1)

        self.host_array = host_array
        self.split_host_array_dict = split_host_array_dict
	
    
    def copy_to_host_array(self):
        for str_f in self.str_fs:
            self.split_host_array_dict[str_f][:] = \
                    self.mainf.get(str_f)[self.slice_xyz]


    def get_event(self):
        evt = self.mainf.enqueue(self.copy_to_host_array)

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

        for strf in str_fs:
            strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
            common.check_value('str_f', strf, strf_list)

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], fields.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

        # global variables and functions
        self.mainf = fields
        self.str_fs = str_fs
        self.slice_xyz = common.slices_two_points(pt0, pt1)
        self.shape = common.shape_two_points(pt0, pt1, len(str_fs))
        self.is_overwrite = is_overwrite

        if is_array:
            self.func = self.set_fields_spatial_value
        else:
            self.func = self.set_fields_single_value


    def set_fields_spatial_value(self, value):
        common.check_value('value.dtype', value.dtype, self.mainf.dtype)
        common.check_value('value.shape', value.shape, [self.shape])
        split_value = np.split(value, len(self.str_fs))

        for str_f, ndarr in zip(self.str_fs, split_value):
            if self.is_overwrite:
                self.mainf.get(str_f)[self.slice_xyz] = ndarr[:]
            else:
                self.mainf.get(str_f)[self.slice_xyz] += ndarr[:]


    def set_fields_single_value(self, value):
        for str_f in self.str_fs:
            if self.is_overwrite:
                self.mainf.get(str_f)[self.slice_xyz] = self.mainf.dtype(value)
            else:
                self.mainf.get(str_f)[self.slice_xyz] += self.mainf.dtype(value)


    def set_fields(self, values, wait_list=[]):
        self.mainf.enqueue(self.func, [values], wait_list)
