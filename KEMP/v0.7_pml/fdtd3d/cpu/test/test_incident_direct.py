import numpy as np
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_random
from kemp.fdtd3d.cpu import IncidentDirect, Fields, GetFields


class TestIncidentDirect(unittest.TestCase):
    def __init__(self, args):
        super(TestIncidentDirect, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1, is_array = self.args

        slices = common.slices_two_points(pt0, pt1)

        # generate random source
        if is_array:
            shape = common.shape_two_points(pt0, pt1)
            value = np.random.rand(*shape).astype(np.float32)
        else:
            value = np.random.ranf()

        # instance
        fields = Fields(nx, ny, nz, '', 'single', 0)

        tfunc = lambda tstep: np.sin(0.03*tstep)
        incident = IncidentDirect(fields, str_f, pt0, pt1, tfunc, value) 

        # host allocations
        eh = np.zeros(fields.ns_pitch, dtype=fields.dtype)
        getf = GetFields(fields, str_f, pt0, pt1)

        # verify
        eh[slices] = fields.dtype(value) * fields.dtype(tfunc(1))
        fields.update_e()
        fields.update_h()
        fields.enqueue_barrier()

        original = eh[slices]
        getf.get_event().wait()
        copy = getf.get_fields()
        norm = np.linalg.norm(original - copy)
        self.assertEqual(norm, 0, '%s, %g' % (self.args, norm))



if __name__ == '__main__':
    ns = nx, ny, nz = 40, 50, 60
    suite = unittest.TestSuite() 
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), False) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), True) ))

    # random sets
    args_list = [(nx, ny, nz, str_f, pt0, pt1, is_array) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in common_random.set_two_points(shape, nx, ny, nz) \
            for is_array in [False, True] ]
    suite.addTests(TestIncidentDirect(args) for args in args_list) 

    unittest.TextTestRunner().run(suite) 
