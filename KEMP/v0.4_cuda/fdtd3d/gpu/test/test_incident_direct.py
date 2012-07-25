import numpy as np
import pycuda.driver as cuda
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_random
from kemp.fdtd3d.gpu import IncidentDirect, Fields


class TestIncidentDirect(unittest.TestCase):
    def __init__(self, args):
        super(TestIncidentDirect, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1, is_array = self.args
        slice_xyz = common.slices_two_points(pt0, pt1)

        # generate random source
        if is_array:
            shape = common.shape_two_points(pt0, pt1)
            value = np.random.rand(*shape).astype(np.float32)
        else:
            value = np.random.ranf()

        # instance
        fields = Fields(0, nx, ny, nz, '', 'single')

        tfunc = lambda tstep: np.sin(0.03*tstep)
        incident = IncidentDirect(fields, str_f, pt0, pt1, tfunc, value) 

        # host allocations
        eh = np.zeros(fields.ns_pitch, dtype=fields.dtype)

        # verify
        eh[slice_xyz] = fields.dtype(value) * fields.dtype(tfunc(1))
        fields.update_e()
        fields.update_h()

        copy_eh_buf = fields.get_buf(str_f)
        copy_eh = np.zeros_like(eh)
        cuda.memcpy_dtoh(copy_eh, copy_eh_buf)

        original = eh[slice_xyz]
        copy = copy_eh[slice_xyz]
        norm = np.linalg.norm(original - copy)
        self.assertEqual(norm, 0, '%s, %g' % (self.args, norm))

        fields.context_pop()


if __name__ == '__main__':
    ns = nx, ny, nz = 40, 50, 60
    suite = unittest.TestSuite() 
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 1, 2), (0, 1, nz-3), False) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (9, 0, 0), (9, 49, 59), True) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 0, 0), (39, 49, 59), True) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 27, 22), (29, 47, 53), True) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (4, 8, 44), (4, 8, 44), False) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (4, 8, 44), (4, 8, 44), True) ))

    # random sets
    args_list = [(nx, ny, nz, str_f, pt0, pt1, is_array) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in common_random.set_two_points(shape, nx, ny, nz) \
            for is_array in [False, True] ]
    test_size = int( len(args_list)*0.1 )
    test_list = [args_list.pop( np.random.randint(len(args_list)) ) for i in xrange(test_size)]
    suite.addTests(TestIncidentDirect(args) for args in test_list) 

    unittest.TextTestRunner().run(suite) 
