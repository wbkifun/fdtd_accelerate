import numpy as np

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu, cpu

from fields import Fields


class GetFields:
    def __init__(self, fields, str_f, pt0, pt1):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_type('str_f', str_f, (str, list, tuple), str)
        common.check_type('pt0', pt0, (list, tuple), int)
        common.check_type('pt1', pt1, (list, tuple), int)

        # local variables
        nodef = fields
        str_fs = common.convert_to_tuple(str_f)
        mainf_list = nodef.mainf_list
        anx = nodef.accum_nx_list

        for strf in str_fs:
            strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
            common.check_value('str_f', strf, strf_list)

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], nodef.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

        # allocation
        shape = common.shape_two_points(pt0, pt1, len(str_fs))
        dummied_shape = common.shape_two_points(pt0, pt1, is_dummy=True)
        host_array = np.zeros(shape, dtype=nodef.dtype)

        split_host_array = np.split(host_array, len(str_fs))
        split_host_array_dict = dict( zip(str_fs, split_host_array) ) 

        getf_list = []
        slices_list = []
        self.gpu, self.cpu = gpu, cpu
        for i, mainf in enumerate(mainf_list):
            nx0 = anx[i]
            nx1 = anx[i+1]-1 if i < len(mainf_list)-1 else anx[i+1]
            overlap = common.intersection_two_lines((nx0, nx1), (pt0[0], pt1[0]))

            if overlap != None:
                x0, y0, z0 = pt0
                x1, y1, z1 = pt1

                slice_pt0 = (overlap[0]-x0, 0, 0)
                slice_pt1 = (overlap[1]-x0, y1-y0, z1-z0)
                slices = []
                for j, p0, p1 in zip([0, 1, 2], slice_pt0, slice_pt1):
                    if dummied_shape[j] != 1:
                        slices.append( slice(p0, p1+1) ) 
                slices_list.append(slices if slices!=[] else [slice(0, 1)] )

                local_pt0 = (overlap[0]-nx0, y0, z0)
                local_pt1 = (overlap[1]-nx0, y1, z1)
                getf_list.append( getattr(self, mainf.device_type). \
                        GetFields(mainf, str_fs, local_pt0, local_pt1) )

        # global variables
        self.str_fs = str_fs
        self.host_array = host_array
        self.split_host_array_dict = split_host_array_dict
        self.getf_list = getf_list
        self.slices_list = slices_list


    def wait(self):
        for getf, slices in zip(self.getf_list, self.slices_list):
            for str_f in self.str_fs:
                getf.get_event().wait()
                self.split_host_array_dict[str_f][slices] = getf.get_fields(str_f)


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
        common.check_type('pt0', pt0, (list, tuple), int)
        common.check_type('pt1', pt1, (list, tuple), int)
        common.check_type('is_array', is_array, bool)
        common.check_type('is_overwrite', is_overwrite, bool)

        # local variables
        nodef = fields
        str_fs = common.convert_to_tuple(str_f)
        mainf_list = nodef.mainf_list
        anx = nodef.accum_nx_list

        for strf in str_fs:
            strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
            common.check_value('str_f', strf, strf_list)

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], nodef.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

        # allocation
        dummied_shape = common.shape_two_points(pt0, pt1, is_dummy=True)

        setf_list = []
        slices_list = []
        self.gpu, self.cpu = gpu, cpu
        for i, mainf in enumerate(mainf_list):
            nx0 = anx[i]
            nx1 = anx[i+1]
            overlap = common.intersection_two_lines((nx0, nx1), (pt0[0], pt1[0]))

            if overlap != None:
                x0, y0, z0 = pt0
                x1, y1, z1 = pt1

                slice_pt0 = (overlap[0]-x0, 0, 0)
                slice_pt1 = (overlap[1]-x0, y1-y0, z1-z0)
                slices = []
                for j, p0, p1 in zip([0, 1, 2], slice_pt0, slice_pt1):
                    if dummied_shape[j] != 1:
                        slices.append( slice(p0, p1+1) ) 
                slices_list.append(slices if slices!=[] else [slice(0, 1)] )

                local_pt0 = (overlap[0]-nx0, y0, z0)
                local_pt1 = (overlap[1]-nx0, y1, z1)
                setf_list.append( getattr(self, mainf.device_type). \
                        SetFields(mainf, str_fs, local_pt0, local_pt1, \
                        is_array, is_overwrite) )

        # global variables and functions
        self.str_fs = str_fs
        self.dtype = nodef.dtype
        self.shape = common.shape_two_points(pt0, pt1, len(str_fs))
        self.setf_list = setf_list
        self.slices_list = slices_list

        if is_array:
            self.set_fields = self.set_fields_spatial_value
        else:
            self.set_fields = self.set_fields_single_value


    def set_fields_spatial_value(self, value):
        common.check_value('value.dtype', value.dtype, self.dtype)
        common.check_value('value.shape', value.shape, [self.shape])
        split_value = np.split(value, len(self.str_fs))

        for setf, slices in zip(self.setf_list, self.slices_list):
            val = np.concatenate([arr[slices] for arr in split_value])
            if val.shape != (1,):
                val = val.reshape([i for i in val.shape if i != 1])
            setf.set_fields(val)


    def set_fields_single_value(self, value):
        for setf in self.setf_list:
            setf.set_fields( self.dtype(value) )
