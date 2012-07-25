import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_update
from kemp.fdtd3d.gpu import Core, Fields


class TestCore(unittest.TestCase):
    def __init__(self, args): 
        super(TestCore, self).__init__() 
        self.args = args


    def runTest(self):
        ufunc, nx, ny, nz, coeff_use, precision_float, tmax = self.args

        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)
        device = gpu_devices[0]
        fields = Fields(context, device, nx, ny, nz, coeff_use, precision_float)
        core = Core(fields)

        # allocations
        ns = fields.ns
        dtype = fields.dtype
        strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']

        ehs = common_update.generate_random_ehs(nx, ny, nz, dtype, ufunc)
        fields.set_eh_bufs(*ehs)

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

            for strf, eh in zip(strf_list, ehs)[:3]:
                cl.enqueue_copy(fields.queue, tmpf, fields.get_buf(strf))
                norm = np.linalg.norm(eh - tmpf[:,:,fields.slice_z])
                max_diff = np.abs(eh - tmpf[:,:,fields.slice_z]).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % (self.args, strf, norm, max_diff) )
                
                if fields.pad != 0:
                    if strf == 'ez':
                        norm2 = np.linalg.norm(tmpf[:,:,-fields.pad:])
                    else:
                        norm2 = np.linalg.norm(tmpf[:,:,-fields.pad-1:])
                    self.assertEqual(norm2, 0, '%s, %s, %g, padding' % (self.args, strf, norm2) )


        elif ufunc == 'h':
            for tstep in xrange(0, tmax):
                fields.update_h()
                common_update.update_h(ehs, chs)

            for strf, eh in zip(strf_list, ehs)[3:]:
                cl.enqueue_copy(fields.queue, tmpf, fields.get_buf(strf))
                norm = np.linalg.norm(eh - tmpf[:,:,fields.slice_z])
                max_diff = np.abs(eh - tmpf[:,:,fields.slice_z]).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % (self.args, strf, norm, max_diff) )

                if fields.pad != 0:
                    if strf == 'hz':
                        norm2 = np.linalg.norm(tmpf[:,:,-fields.pad:])
                    else:
                        norm2 = np.linalg.norm(tmpf[:,:,-fields.pad:])
                    self.assertEqual(norm2, 0, '%s, %s, %g, padding' % (self.args, strf, norm2) )


if __name__ == '__main__':
    nx, ny = 40, 50
    suite = unittest.TestSuite() 
    suite.addTest(TestCore( ('e', nx, ny, 60, 'e', 'single', 1) ))
    suite.addTest(TestCore( ('h', nx, ny, 60, '', 'single', 1) ))

    args_list = [ \
            (ufunc, nx, ny, nz, coeff_use, precision_float, 1) \
            for ufunc in ['e', 'h'] \
            for nz in range(56, 73) \
            for coeff_use in ['', 'e', 'h'] \
            for precision_float in ['single', 'double'] ]
    test_size = int( len(args_list)*0.1 )
    test_list = [args_list.pop( np.random.randint(len(args_list)) ) for i in xrange(test_size)]
    suite.addTests(TestCore(args) for args in test_list) 

    suite.addTest(TestCore( ('e', nx, ny, 60, 'e', 'single', 10) ))
    suite.addTest(TestCore( ('e', nx, ny, 60, 'e', 'double', 10) ))

    unittest.TextTestRunner().run(suite) 
