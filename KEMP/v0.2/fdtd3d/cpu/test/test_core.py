import numpy as np
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_update
from kemp.fdtd3d.cpu import Core, Fields


class TestCore(unittest.TestCase):
    def __init__(self, args):
        super(TestCore, self).__init__()
        self.args = args


    def runTest(self):
        ufunc, nx, ny, nz, coeff_use, precision_float, use_cpu_core, split, tmax = self.args
        fields = Fields(nx, ny, nz, coeff_use, precision_float, use_cpu_core)
        core = Core(fields)

        strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
        slice_xyz = [slice(None, None), slice(None, None), fields.slice_z]

        # allocations
        ns = fields.ns
        dtype = fields.dtype

        ehs = common_update.generate_random_ehs(nx, ny, nz, dtype, ufunc)
        fields.set_ehs(*ehs)

        ces, chs = common_update.generate_random_cs(coeff_use, nx, ny, nz, dtype)
        if 'e' in coeff_use:
            fields.set_ces(*ces)
        if 'h' in coeff_use:
            fields.set_chs(*chs)

        # update
        if ufunc == 'e':
            for tstep in xrange(0, tmax):
                fields.update_e()
                common_update.update_e(ehs, ces)
            fields.enqueue_barrier()

            for strf, eh in zip(strf_list, ehs)[:3]:
                norm = np.linalg.norm(eh - fields.get(strf)[slice_xyz])
                max_diff = np.abs(eh - fields.get(strf)[slice_xyz]).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % (self.args, strf, norm, max_diff) )

                if fields.pad != 0:
                    if strf == 'ez':
                        norm2 = np.linalg.norm(fields.get(strf)[:,:,-fields.pad:])
                    else:
                        norm2 = np.linalg.norm(fields.get(strf)[:,:,-fields.pad-1:])
                    self.assertEqual(norm2, 0, '%s, %s, %g, padding' % (self.args, strf, norm2) )

        elif ufunc == 'h':
            for tstep in xrange(0, tmax):
                fields.update_h()
                common_update.update_h(ehs, chs)
            fields.enqueue_barrier()

            for strf, eh in zip(strf_list, ehs)[3:]:
                norm = np.linalg.norm(eh - fields.get(strf)[slice_xyz])
                max_diff = np.abs(eh - fields.get(strf)[slice_xyz]).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % \
                        (self.args, strf, norm, max_diff) )

                if fields.pad != 0:
                    if strf == 'hz':
                        norm2 = np.linalg.norm(fields.get(strf)[:,:,-fields.pad:])
                    else:
                        norm2 = np.linalg.norm(fields.get(strf)[:,:,-fields.pad:])
                    self.assertEqual(norm2, 0, '%s, %s, %g, padding' % (self.args, strf, norm2) )



if __name__ == '__main__':
    suite = unittest.TestSuite() 

    nx, ny = 40, 50
    suite.addTest(TestCore( ('e', nx, ny, 61, '', 'single', 0, '', 1) ))

    args_list = [ \
            (ufunc, nx, ny, nz, coeff_use, precision_float, use_cpu_core, split, 1) \
            for ufunc in ['e', 'h'] \
            for nz in [61, 62, 63, 64] \
            for coeff_use in ['', 'e', 'h'] \
            for precision_float in ['single', 'double'] \
            for use_cpu_core in [0, 1] \
            for split in ['', 'e', 'h'] ]
    test_size = int( len(args_list)*0.1 )
    test_list = [args_list.pop( np.random.randint(len(args_list)) ) for i in xrange(test_size)]
    suite.addTests(TestCore(args) for args in test_list) 

    suite.addTest(TestCore( ('e', nx, ny, 64, 'e', 'single', 0, '', 10) ))
    suite.addTest(TestCore( ('e', nx, ny, 64, 'e', 'double', 0, '', 10) ))

    unittest.TextTestRunner().run(suite) 
