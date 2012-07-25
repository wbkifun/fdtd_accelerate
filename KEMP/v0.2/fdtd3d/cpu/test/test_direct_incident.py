import numpy as np
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common
from kemp.fdtd3d.util.common_test import random_set_two_points
from kemp.fdtd3d.cpu import DirectIncident, Fields, GetFields


class TestDirectIncident(unittest.TestCase):
    def __init__(self, args):
        super(TestDirectIncident, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1, is_array, mpi_type = self.args

        slices = common.slice_index_two_points(pt0, pt1)

        # generate random source
        if is_array:
            shape = common.shape_two_points(pt0, pt1)
            value = np.random.rand(*shape).astype(np.float32)
        else:
            value = np.random.ranf()

        # instance
        fields = Fields(nx, ny, nz, '', 'single', 0, mpi_type=mpi_type)

        tfunc = lambda tstep: np.sin(0.03*tstep)
        incident = DirectIncident(fields, str_f, pt0, pt1, tfunc, value) 

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
    suite.addTest(TestDirectIncident( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), False, '') ))
    suite.addTest(TestDirectIncident( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), True, '') ))
    suite.addTest(TestDirectIncident( (3, ny, nz, 'ex', (0, 33, 47), (2, 33, 47), True, 'x+') ))

    # random sets
    args_list = [(nx, ny, nz, str_f, pt0, pt1, is_array, '') \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in random_set_two_points(shape, nx, ny, nz) \
            for is_array in [False, True] ]
    suite.addTests(TestDirectIncident(args) for args in args_list) 

    # random sets with mpi_type
    args_list = [(3, ny, nz, str_f, pt0, pt1, is_array, mpi_type) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in random_set_two_points(shape, 3, ny, nz) \
            for is_array in [False, True] \
            for mpi_type in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-'] ]
    suite.addTests(TestDirectIncident(args) for args in args_list) 

    unittest.TextTestRunner().run(suite) 
