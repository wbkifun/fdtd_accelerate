import numpy as np

from kemp.fdtd3d.util import common


class Fields(object):
    def __init__(self, nx, ny, nz, precision_float='single', segment_nbytes=16):
        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_type('segment_nbytes', segment_nbytes, int)
        common.check_value('precision_float', precision_float, ('single', 'double'))

        # local variables
        dtype = {'single':np.float32, 'double':np.float64}[precision_float]

        # padding for the nz which is multi of segment size
        align_size = segment_nbytes / np.nbytes[dtype]
        pad = int(np.ceil(float(nz) / align_size) * align_size) - nz
        slice_z = slice(None, None) if pad == 0 else slice(None, -pad)
        nz_pitch = nz + pad

        ns = [nx, ny, nz]
        ns_pitch = [nx, ny, nz_pitch]
        ns_pad = [nx, ny, pad]

        # allocations
        ehs = [np.zeros(ns_pitch, dtype) for i in range(6)]
        ces = [np.ones(ns_pitch, dtype)*0.5 for i in range(3)]
        chs = [np.ones(ns_pitch, dtype)*0.5 for i in range(3)]

        # global variables
        self.dx = 1.
        self.dt = 0.5
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
        self.dtype = dtype

        self.ehs = ehs
        self.ex, self.ey, self.ez = ehs[:3]
        self.hx, self.hy, self.hz = ehs[3:]
        self.ces = ces
        self.cex, self.cey, self.cez = ces
        self.chs = chs
        self.chx, self.chy, self.chz = chs

        # update list
        self.instance_list = []
        self.append_instance = lambda instance: \
                common.append_instance(self.instance_list, instance)


    def get(self, str_f):
        value_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz', 'cex', 'cey', 'cez', 'chx', 'chy', 'chz']
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

        self.cex[:,:,self.slice_z] = cex[:]
        self.cey[:,:,self.slice_z] = cey[:]
        self.cez[:,:,self.slice_z] = cez[:]


    def set_chs(self, chx, chy, chz):
        common.check_type('chx', chx, np.ndarray)
        common.check_type('chy', chy, np.ndarray)
        common.check_type('chz', chz, np.ndarray)

        self.chx[:,:,self.slice_z] = chx[:]
        self.chy[:,:,self.slice_z] = chy[:]
        self.chz[:,:,self.slice_z] = chz[:]


    def update_e(self):
        for instance in self.instance_list:
            instance.update_e()


    def update_h(self):
        for instance in self.instance_list:
            instance.update_h()
