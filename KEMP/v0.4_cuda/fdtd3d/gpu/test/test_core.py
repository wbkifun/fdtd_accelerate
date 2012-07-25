import numpy as np
import pycuda.driver as cuda
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_random
from kemp.fdtd3d import gpu, naive


class TestCore(unittest.TestCase):
    def __init__(self, args): 
        super(TestCore, self).__init__() 
        self.args = args


    def runTest(self):
        ufunc, nx, ny, nz, coeff_use, precision_float, tmax = self.args

        fields = gpu.Fields(0, nx, ny, nz, coeff_use, precision_float)
        gpu.Core(fields)

        fields_ref = naive.Fields(nx, ny, nz, precision_float, segment_nbytes=64)
        naive.Core(fields_ref)

        # allocations
        ns = fields.ns
        dtype = fields.dtype
        strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']

        ehs = common_random.generate_ehs(nx, ny, nz, dtype, ufunc)
        fields.set_eh_bufs(*ehs)
        fields_ref.set_ehs(*ehs)

        ces, chs = common_random.generate_cs(nx, ny, nz, dtype, coeff_use)
        if 'e' in coeff_use:
            fields.set_ce_bufs(*ces)
            fields_ref.set_ces(*ces)
        if 'h' in coeff_use:
            fields.set_ch_bufs(*chs)
            fields_ref.set_chs(*chs)

        tmpf = np.zeros(fields.ns_pitch, dtype=dtype)

        # update
        if ufunc == 'e':
            for tstep in xrange(0, tmax):
                fields.update_e()
                fields_ref.update_e()

            for strf, eh in zip(strf_list, ehs)[:3]:
                cuda.memcpy_dtoh(tmpf, fields.get_buf(strf))
                norm = np.linalg.norm(fields_ref.get(strf) - tmpf)
                max_diff = np.abs(fields_ref.get(strf) - tmpf).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % (self.args, strf, norm, max_diff) )

        elif ufunc == 'h':
            for tstep in xrange(0, tmax):
                fields.update_h()
                fields_ref.update_h()

            for strf, eh in zip(strf_list, ehs)[3:]:
                cuda.memcpy_dtoh(tmpf, fields.get_buf(strf))
                norm = np.linalg.norm(fields_ref.get(strf) - tmpf)
                max_diff = np.abs(fields_ref.get(strf) - tmpf).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % (self.args, strf, norm, max_diff) )

        fields.context.pop()



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
