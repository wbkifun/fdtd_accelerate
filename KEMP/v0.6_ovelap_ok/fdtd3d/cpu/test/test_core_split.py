import numpy as np
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_random
from kemp.fdtd3d import cpu, naive


class TestCore(unittest.TestCase):
    def __init__(self, args):
        super(TestCore, self).__init__()
        self.args = args


    def runTest(self):
        ufunc, nx, ny, nz, coeff_use, precision_float, use_cpu_core, tmax = self.args
        qtask = cpu.QueueTask()
        fields = cpu.Fields(qtask, nx, ny, nz, coeff_use, precision_float, use_cpu_core)
        cpu.Core(fields)

        fields_ref = naive.Fields(nx, ny, nz, precision_float, segment_nbytes=16)
        naive.Core(fields_ref)

        # allocations
        ns = fields.ns
        dtype = fields.dtype

        ehs = common_random.generate_ehs(nx, ny, nz, dtype, ufunc)
        fields.set_ehs(*ehs)
        fields_ref.set_ehs(*ehs)

        ces, chs = common_random.generate_cs(nx, ny, nz, dtype, coeff_use)
        if 'e' in coeff_use:
            fields.set_ces(*ces)
            fields_ref.set_ces(*ces)
        if 'h' in coeff_use:
            fields.set_chs(*chs)
            fields_ref.set_chs(*chs)

        # verify
        strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']

        if ufunc == 'e':
            for tstep in xrange(0, tmax):
                for instance in fields.instance_list:
                    instance.update_e('pre')
                    instance.update_e('post')
                fields_ref.update_e()
            fields.enqueue_barrier()

            for strf, eh in zip(strf_list, ehs)[:3]:
                norm = np.linalg.norm(fields.get(strf) - fields_ref.get(strf))
                max_diff = np.abs(fields.get(strf) - fields_ref.get(strf)).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % (self.args, strf, norm, max_diff) )

        elif ufunc == 'h':
            for tstep in xrange(0, tmax):
                for instance in fields.instance_list:
                    instance.update_h('pre')
                    instance.update_h('post')
                fields_ref.update_h()
            fields.enqueue_barrier()

            for strf, eh in zip(strf_list, ehs)[3:]:
                norm = np.linalg.norm(fields.get(strf) - fields_ref.get(strf))
                max_diff = np.abs(fields.get(strf) - fields_ref.get(strf)).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % \
                        (self.args, strf, norm, max_diff) )



if __name__ == '__main__':
    suite = unittest.TestSuite() 

    nx, ny = 3, 50
    suite.addTest(TestCore( ('e', nx, ny, 64, '', 'single', 0, 1) ))

    args_list = [ \
            (ufunc, nx, ny, nz, coeff_use, precision_float, use_cpu_core, 1) \
            for ufunc in ['e', 'h'] \
            for nz in [61, 62, 63, 64] \
            for coeff_use in ['', 'e', 'h'] \
            for precision_float in ['single', 'double'] \
            for use_cpu_core in [0, 1] ]
    test_size = int( len(args_list)*0.1 )
    test_list = [args_list.pop( np.random.randint(len(args_list)) ) for i in xrange(test_size)]
    suite.addTests(TestCore(args) for args in test_list) 

    suite.addTest(TestCore( ('e', nx, ny, 64, 'e', 'single', 0, 10) ))
    suite.addTest(TestCore( ('e', nx, ny, 64, 'e', 'double', 0, 10) ))

    unittest.TextTestRunner().run(suite) 
