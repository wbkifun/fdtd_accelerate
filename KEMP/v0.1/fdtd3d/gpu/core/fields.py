import numpy as np
import pyopencl as cl
import unittest

from kemp.fdtd3d.util import common, common_gpu
from kemp.fdtd3d.test import common_update


class Fields:
    def __init__(self, context, device, \
            nx, ny, nz, \
            coeff_use='e', \
            precision_float='single', \
            local_work_size=256, \
            global_work_size=0):
        """
        """

        common.check_type('context', context, cl.Context)
        common.check_type('device', device, cl.Device)
        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_type('global_work_size', global_work_size, int)
        common.check_type('local_work_size', local_work_size, int)

        common.check_value('coeff_use', coeff_use, ('', 'e', 'h', 'eh'))
        common.check_value('precision_float', precision_float, ('single', 'double'))

        self.context = context
        self.device = device
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ls = local_work_size
        self.gs = global_work_size
        self.coeff_use = coeff_use
        self.dtype = {'single':np.float32, 'double':np.float64}[precision_float]
        self.dtype_str = {'single':'float', 'double':'double'}[precision_float]
        self.dtype_str_list = { \
                'single':['float', ''], \
                'double':['double', '#pragma OPENCL EXTENSION cl_khr_fp64 : enable'] }[precision_float]

        self.device_type = 'gpu'

        # padding for the nz which is multiple of 16 (float32) or 8 (float64)
        self.align_size = a_size = {'single':16, 'double':8}[precision_float]  # 64 Bytes
        self.pad = pad = int(np.ceil(float(nz) / a_size) * a_size) - nz
        self.slz = slice(None, None) if pad == 0 else slice(None, -pad)
        self.nz_pitch = nz_pitch = nz + pad

        self.dtype_str_list.append( '' if pad==0 else '-%s' % pad )

        # ns, queue, global_size
        self.ns = [np.int32(nx), np.int32(ny), np.int32(nz)]
        self.ns_pitch = [np.int32(nx), np.int32(ny), np.int32(nz_pitch)]
        self.ns_pad = [np.int32(nx), np.int32(ny), np.int32(pad)]
        self.queue = cl.CommandQueue(self.context, self.device)
        if self.gs == 0:
            self.gs = common_gpu.get_optimal_gs(self.device)

        # on/off the coefficient arrays
        self.ce_on = True if 'e' in self.coeff_use else False
        self.ch_on = True if 'h' in self.coeff_use else False

        # allocations
        f = np.zeros(self.ns_pitch, dtype=self.dtype)
        cf = np.ones_like(f) * 0.5
        mf = cl.mem_flags

        self.eh_bufs = [cl.Buffer(self.context, mf.READ_WRITE, f.nbytes) \
                for i in range(6)]
        for eh_buf in self.eh_bufs:
            cl.enqueue_copy(self.queue, eh_buf, f) 
        self.ex_buf, self.ey_buf, self.ez_buf = self.eh_bufs[:3]
        self.hx_buf, self.hy_buf, self.hz_buf = self.eh_bufs[3:]

        if self.ce_on:
            self.ce_bufs = [cl.Buffer(self.context, mf.READ_ONLY, cf.nbytes) \
                    for i in range(3)]
            self.cex_buf, self.cey_buf, self.cez_buf = self.ce_bufs

        if self.ch_on:
            self.ch_bufs = [cl.Buffer(self.context, mf.READ_ONLY, cf.nbytes) \
                    for i in range(3)]
            self.chx_buf, self.chy_buf, self.chz_buf = self.ch_bufs

        del f, cf

        # program
        macros = ['ARGS_CE', 'CEX', 'CEY', 'CEZ', \
                'ARGS_CH', 'CHX', 'CHY', 'CHZ', \
                'DX', 'DTYPE', 'PRAGMA_fp64', 'PAD']

        values = ['', '0.5', '0.5', '0.5', \
                '', '0.5', '0.5', '0.5', \
                str(self.ls)] + self.dtype_str_list

        self.e_args = self.ns_pitch + self.eh_bufs
        self.h_args = self.ns_pitch + self.eh_bufs

        if self.ce_on:
            values[:4] = [ \
                    ', __global DTYPE *cex, __global DTYPE *cey, __global DTYPE *cez', \
                    'cex[idx]', 'cey[idx]', 'cez[idx]']
            self.e_args += self.ce_bufs

        if self.ch_on:
            values[4:8] = [ \
                    ', __global DTYPE *chx, __global DTYPE *chy, __global DTYPE *chz', \
                    'chx[idx]', 'chy[idx]', 'chz[idx]']
            self.h_args += self.ch_bufs

        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'core.cl').read(), macros, values)
        self.program = cl.Program(self.context, ksrc).build()


    def get_buf(self, str_f):
        return self.__dict__[str_f + '_buf']


    def set_ce_bufs(self, cex, cey, cez):
        if self.ce_on:
            pad_arr = np.zeros(self.ns_pad, dtype=self.dtype)
            for ce_buf, ce in zip(self.ce_bufs, [cex, cey, cez]):
                ce_pitch = np.append(ce, pad_arr, 2).copy('C')
                cl.enqueue_copy(self.queue, ce_buf, ce_pitch)
        else:
            raise AttributeError("The Fields instance has no ce buffer arrays. You should add 'e' in the option 'coeff_use'")


    def set_ch_bufs(self, chx, chy, chz):
        if self.ch_on:
            pad_arr = np.zeros(self.ns_pad, dtype=self.dtype)
            for ch_buf, ch in zip(self.ch_bufs, [chx, chy, chz]):
                ch_pitch = np.append(ch, pad_arr, 2).copy('C')
                cl.enqueue_copy(self.queue, ch_buf, ch_pitch)
        else:
            raise AttributeError("The Fields instance has no ch buffer arrays. You should add 'h' in the option 'coeff_use'")


    def update_e(self):
        self.program.update_e(self.queue, (self.gs,), (self.ls,), *self.e_args)


    def update_h(self):
        self.program.update_h(self.queue, (self.gs,), (self.ls,), *self.h_args)




class TestFields(unittest.TestCase):
    def __init__(self, args): 
        super(TestFields, self).__init__() 
        self.args = args


    def runTest(self):
        ufunc, nx, ny, nz, coeff_use, precision_float, tmax = self.args

        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)
        device = gpu_devices[0]
        fields = Fields(context, device, nx, ny, nz, coeff_use, precision_float)

        # allocations
        ns = fields.ns
        dtype = fields.dtype
        strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']

        ehs = common_update.generate_random_ehs(ufunc, nx, ny, nz, dtype)
        pad_arr = np.zeros(fields.ns_pad, dtype=dtype)
        for strf, eh in zip(strf_list, ehs):
            eh_pitch = np.append(eh, pad_arr, 2).copy('C')
            cl.enqueue_copy(fields.queue, fields.get_buf(strf), eh_pitch)

        ces, chs = common_update.generate_random_cs(coeff_use, nx, ny, nz, dtype)
        if 'e' in coeff_use:
            fields.set_ce_bufs(*ces)
        if 'h' in coeff_use:
            fields.set_ch_bufs(*chs)

        tmpf = np.zeros(fields.ns_pitch, dtype=dtype)

        # update
        if ufunc == 'e':
            for tstep in xrange(0, tmax):
                fields.update_e()
                common_update.update_e(ehs, ces)

            for strf, eh in zip(strf_list, ehs):
                cl.enqueue_copy(fields.queue, tmpf, fields.get_buf(strf))
                norm = np.linalg.norm(eh - tmpf[:,:,fields.slz])
                max_diff = np.abs(eh - tmpf[:,:,fields.slz]).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % \
                        (self.args, strf, norm, max_diff) )

        elif ufunc == 'h':
            for tstep in xrange(0, tmax):
                fields.update_h()
                common_update.update_h(ehs, chs)

            for strf, eh in zip(strf_list, ehs):
                cl.enqueue_copy(fields.queue, tmpf, fields.get_buf(strf))
                norm = np.linalg.norm(eh - tmpf[:,:,fields.slz])
                max_diff = np.abs(eh - tmpf[:,:,fields.slz]).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % \
                        (self.args, strf, norm, max_diff) )



if __name__ == '__main__':
    nx, ny = 40, 50
    suite = unittest.TestSuite() 
    #suite.addTest(TestFields( ('e', nx, ny, 64, 'e', 'single', 2) ))
    #suite.addTest(TestFields( ('h', nx, ny, 64, '', 'single', 1) ))

    args_list = [ \
            (ufunc, nx, ny, nz, coeff_use, precision_float, 1) \
            for ufunc in ['e', 'h'] \
            for nz in range(56, 73) \
            for coeff_use in ['', 'e', 'h'] \
            for precision_float in ['single', 'double'] ]
    suite.addTests(TestFields(args) for args in args_list) 

    suite.addTest(TestFields( ('e', nx, ny, 64, 'e', 'single', 10) ))
    suite.addTest(TestFields( ('e', nx, ny, 64, 'e', 'double', 10) ))

    unittest.TextTestRunner().run(suite) 
