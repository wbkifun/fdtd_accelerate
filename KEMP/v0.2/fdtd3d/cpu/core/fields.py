import numpy as np

from kemp.fdtd3d.util import common, common_cpu
from queue_task import QueueTask


class Fields:
    def __init__(self, nx, ny, nz, \
            coeff_use='', \
            precision_float='single', \
            use_cpu_core=0, \
            mpi_type=''):
        """
        """

        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_value('coeff_use', coeff_use, ('', 'e', 'h', 'eh'))
        common.check_value('precision_float', precision_float, \
                ('single', 'double'))
        common.check_type('use_cpu_core', use_cpu_core, int)
        common.check_value('mpi_type', mpi_type, \
                ('', 'x+', 'x-', 'y+', 'y-', 'z+', 'z-'))

        # local variables
        dtype = {'single':np.float32, 'double':np.float64}[precision_float]

        # padding for the nz which is multiple of 4 (float32) or 2 (float64)
        align_size = {'single':4, 'double':2}[precision_float]   # 16 Bytes
        pad = int(np.ceil(float(nz) / align_size) * align_size) - nz
        slice_z = slice(None, None) if pad == 0 else slice(None, -pad)
        nz_pitch = nz + pad

        ns = [nx, ny, nz]
        ns_pitch = [nx, ny, nz_pitch]
        ns_pad = [nx, ny, pad]

        # on/off the coefficient arrays
        ce_on = True if 'e' in coeff_use else False
        ch_on = True if 'h' in coeff_use else False

        # allocations
        ehs = [np.zeros(ns_pitch, dtype) for i in range(6)]

        if ce_on:
            ces = [np.ones(ns_pitch, dtype)*0.5 for i in range(3)]

        if ch_on:
            chs = [np.ones(ns_pitch, dtype)*0.5 for i in range(3)]

        # global variables and functions
        self.device_type = 'cpu'
        self.qtask = QueueTask()
        self.enqueue = self.qtask.enqueue
        self.enqueue_barrier = self.qtask.enqueue_barrier

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ns = ns
        self.ns_pitch = ns_pitch
        self.ns_pad = ns_pad

        self.align_size = align_size
        self.pad = pad
        self.slice_z = slice_z

        self.precision_float = precision_float
        self.use_cpu_core = use_cpu_core
        self.dtype = dtype

        self.coeff_use = coeff_use
        self.ce_on = ce_on
        self.ch_on = ch_on

        self.ehs = ehs
        self.ex, self.ey, self.ez = ehs[:3]
        self.hx, self.hy, self.hz = ehs[3:]
        if ce_on:
            self.ces = ces
            self.cex, self.cey, self.cez = ces
        if ch_on:
            self.chs = chs
            self.chx, self.chy, self.chz = chs

        self.mpi_type = mpi_type

        # update list
        self.instance_list = []
        self.append_instance = lambda instance: \
                common.append_instance(self.instance_list, instance)

        split = {'+': 'h', '-': 'e', '': ''}[mpi_type[1:]]
        if split == '':
            self.update_e = self.update_e_whole
            self.update_h = self.update_h_whole

        elif split == 'e':
            self.update_e = self.update_e_split
            self.update_h = self.update_h_whole

        elif split == 'h':
            self.update_e = self.update_e_whole
            self.update_h = self.update_h_split


    def get(self, str_f):
        value_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
        if self.ce_on:
            value_list += ['cex', 'cey', 'cez']
        if self.ch_on:
            value_list += ['chx', 'chy', 'chz']

        common.check_value('str_f', str_f, value_list)

        return self.__dict__[str_f]


    def set_ehs(self, ex, ey, ez, hx, hy, hz):  # for unittest
        eh_list = [ex, ey, ez, hx, hy, hz]
        for eh in eh_list:
            common.check_type('eh', eh, np.ndarray)

        for eh, f in zip(self.ehs, eh_list):
            eh[:,:,self.slice_z] = f[:]


    def set_ces(self, cex, cey, cez):
        common.check_type('cex', cex, np.ndarray)
        common.check_type('cey', cey, np.ndarray)
        common.check_type('cez', cez, np.ndarray)

        if self.ce_on:
            self.cex[:,:,self.slice_z] = cex[:]
            self.cey[:,:,self.slice_z] = cey[:]
            self.cez[:,:,self.slice_z] = cez[:]
        else:
            raise AttributeError("The Fields instance has no ce arrays. You should add 'e' in the option 'coeff_use'")


    def set_chs(self, chx, chy, chz):
        common.check_type('chx', chx, np.ndarray)
        common.check_type('chy', chy, np.ndarray)
        common.check_type('chz', chz, np.ndarray)

        if self.ch_on:
            self.chx[:,:,self.slice_z] = chx[:]
            self.chy[:,:,self.slice_z] = chy[:]
            self.chz[:,:,self.slice_z] = chz[:]
        else:
            raise AttributeError("The Fields instance has no ch arrays. You should add 'h' in the option 'coeff_use'")


    def split_slices_dict(self, eh, slices):
        common.check_value('eh', eh, ('e', 'h'))
        common.check_type('slices', slices, (list, tuple), slice)

        overlap = lambda sl: \
                common.intersection_two_slices(self.ns, slices, \
                (sl, slice(None, None), slice(None, None)) )

        if eh == 'e':
            slices_dict = { \
                    '': slices, \
                    'pre': overlap( slice(-2, None) ), \
                    'mid': overlap( slice(1, -2) ), \
                    'post': overlap( slice(None, 1) ) }
        elif eh == 'h':
            slices_dict = { \
                    '': slices, \
                    'pre': overlap( slice(None, 2) ), \
                    'mid': overlap( slice(2, -1) ), \
                    'post': overlap( slice(-1, None) ) }

        return slices_dict


    def split_points_dict(self, eh, pt0, pt1):
        common.check_value('eh', eh, ('e', 'h'))
        common.check_type('pt0', pt0, (list, tuple), int)
        common.check_type('pt1', pt1, (list, tuple), int)

        overlap = lambda pt2, pt3: \
                common.intersection_two_regions(pt0, pt1, pt2, pt3)

        nx, ny, nz = self.ns

        if eh == 'e':
            points_dict = { \
                    '': (pt0, pt1), \
                    'pre': overlap((nx-2, 0, 0), (nx-1, ny-1, nz-1)), \
                    'mid': overlap((1, 0, 0), (nx-3, ny-1, nz-1)), \
                    'post': overlap((0, 0, 0), (0, ny-1, nz-1)) }
        elif eh == 'h':
            points_dict = { \
                    '': (pt0, pt1), \
                    'pre': overlap((0, 0, 0), (1, ny-1, nz-1)), \
                    'mid': overlap((2, 0, 0), (nx-2, ny-1, nz-1)), \
                    'post': overlap((nx-1, 0, 0), (nx-1, ny-1, nz-1)) }

        return points_dict


    def update_e_whole(self):
        for instance in self.instance_list:
            instance.update_e()


    def update_h_whole(self):
        for instance in self.instance_list:
            instance.update_h()


    def update_e_split(self):
        for part in ['pre', 'mid', 'post']:
            for instance in self.instance_list:
                instance.update_e(part)


    def update_h_split(self):
        for part in ['pre', 'mid', 'post']:
            for instance in self.instance_list:
                instance.update_h(part)
