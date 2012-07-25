import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu


class Fields:
    def __init__(self, context, device, \
            nx, ny, nz, \
            coeff_use='', \
            precision_float='single', \
            local_work_size=256):
        """
        """

        common.check_type('context', context, cl.Context)
        common.check_type('device', device, cl.Device)
        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_value('coeff_use', coeff_use, ('', 'e', 'h', 'eh'))
        common.check_value('precision_float', precision_float, ('single', 'double'))
        common.check_type('local_work_size', local_work_size, int)

        # local variables
        queue = cl.CommandQueue(context, device)
        pragma_fp64 = ''
        if precision_float == 'double':
            extensions = device.get_info(cl.device_info.EXTENSIONS)
            if 'cl_khr_fp64' in extensions:
                pragma_fp64 = '#pragma OPENCL EXTENSION cl_khr_fp64 : enable'
            elif 'cl_amd_fp64' in extensions:
                pragma_fp64 = '#pragma OPENCL EXTENSION cl_amd_fp64 : enable'
            else:
                precision_float = 'single'
                print('Warning: The %s GPU device is not support the double-precision.') % \
                        device.get_info(cl.device_info.NAME)
                print('The precision is changed to \'single\'.')

        dtype = {'single':np.float32, 'double':np.float64}[precision_float]
        dtype_str_list = { \
                'single':['float', ''], \
                'double':['double', pragma_fp64] }[precision_float]

        # padding for the nz which is multiple of 16 (float32) or 8 (float64)
        segment_nbytes = 64
        align_size = segment_nbytes / np.nbytes[dtype]
        pad = int(np.ceil(float(nz) / align_size) * align_size) - nz
        slice_z = slice(None, None) if pad == 0 else slice(None, -pad)
        nz_pitch = nz + pad

        ns = [np.int32(nx), np.int32(ny), np.int32(nz)]
        ns_pitch = [np.int32(nx), np.int32(ny), np.int32(nz_pitch)]
        ns_pad = [np.int32(nx), np.int32(ny), np.int32(pad)]

        # on/off the coefficient arrays
        ce_on = True if 'e' in coeff_use else False
        ch_on = True if 'h' in coeff_use else False

        # allocations
        f = np.zeros(ns_pitch, dtype)
        cf = np.ones_like(f) * 0.5

        mflags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
        eh_bufs = [cl.Buffer(context, mflags, hostbuf=f) for i in range(6)]

        c_mflags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        if ce_on:
            ce_bufs = [cl.Buffer(context, c_mflags, hostbuf=cf) for i in range(3)]

        if ch_on:
            ch_bufs = [cl.Buffer(context, c_mflags, hostbuf=cf) for i in range(3)]

        del f, cf

        # global variables
        self.device_type = 'gpu'
        self.context = context
        self.device = device
        self.queue = queue

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
        self.dtype_str_list = dtype_str_list 

        self.coeff_use = coeff_use
        self.ce_on = ce_on
        self.ch_on = ch_on

        self.eh_bufs = eh_bufs
        self.ex_buf, self.ey_buf, self.ez_buf = eh_bufs[:3]
        self.hx_buf, self.hy_buf, self.hz_buf = eh_bufs[3:]
        if ce_on:
            self.ce_bufs = ce_bufs
            self.cex_buf, self.cey_buf, self.cez_buf = ce_bufs
        if ch_on:
            self.ch_bufs = ch_bufs
            self.chx_buf, self.chy_buf, self.chz_buf = ch_bufs

        self.ls = ls = local_work_size
        nmax = nx * ny * nz_pitch
        remainder = nmax % ls
        self.gs = nmax if remainder == 0 else nmax - remainder + ls 


        # create update list
        self.instance_list = []
        self.append_instance = lambda instance: \
            common.append_instance(self.instance_list, instance)


    def get_buf(self, str_f):
        value_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
        if self.ce_on:
            value_list += ['cex', 'cey', 'cez']
        if self.ch_on:
            value_list += ['chx', 'chy', 'chz']

        common.check_value('str_f', str_f, value_list)

        return self.__dict__[str_f + '_buf']


    def set_eh_bufs(self, ex, ey, ez, hx, hy, hz):  # for unittest
        eh_list = [ex, ey, ez, hx, hy, hz]
        for eh in eh_list:
            common.check_type('eh', eh, np.ndarray)

        pad_arr = np.zeros(self.ns_pad, dtype=self.dtype)
        for f_buf, f in zip(self.eh_bufs, eh_list):
            f_pitch = np.append(f, pad_arr, 2).copy('C')
            cl.enqueue_copy(self.queue, f_buf, f_pitch)


    def set_ce_bufs(self, cex, cey, cez):
        common.check_type('cex', cex, np.ndarray)
        common.check_type('cey', cey, np.ndarray)
        common.check_type('cez', cez, np.ndarray)

        if self.ce_on:
            pad_arr = np.zeros(self.ns_pad, dtype=self.dtype)
            for ce_buf, ce in zip(self.ce_bufs, [cex, cey, cez]):
                ce_pitch = np.append(ce, pad_arr, 2).copy('C')
                cl.enqueue_copy(self.queue, ce_buf, ce_pitch)
        else:
            raise AttributeError("The Fields instance has no ce buffer arrays. You should add 'e' in the option 'coeff_use'")


    def set_ch_bufs(self, chx, chy, chz):
        common.check_type('chx', chx, np.ndarray)
        common.check_type('chy', chy, np.ndarray)
        common.check_type('chz', chz, np.ndarray)

        if self.ch_on:
            pad_arr = np.zeros(self.ns_pad, dtype=self.dtype)
            for ch_buf, ch in zip(self.ch_bufs, [chx, chy, chz]):
                ch_pitch = np.append(ch, pad_arr, 2).copy('C')
                cl.enqueue_copy(self.queue, ch_buf, ch_pitch)
        else:
            raise AttributeError("The Fields instance has no ch buffer arrays. You should add 'h' in the option 'coeff_use'")


    def update_e(self):
        for instance in self.instance_list:
            instance.update_e()


    def update_h(self):
        for instance in self.instance_list:
            instance.update_h()
