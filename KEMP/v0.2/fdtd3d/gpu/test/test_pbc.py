import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu, common_update
from kemp.fdtd3d.gpu import Pbc, Fields, GetFields


class TestPbc(unittest.TestCase):
    def __init__(self, args):
        super(TestPbc, self).__init__()
        self.args = args


    def runTest(self):
        axis, nx, ny, nz, precision_float = self.args

        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)
        device = gpu_devices[0]
        fields = Fields(context, device, nx, ny, nz, '', precision_float)
        pbc = Pbc(fields, axis)

        # allocations
        ehs = common_update.generate_random_ehs(nx, ny, nz, fields.dtype)
        fields.set_eh_bufs(*ehs)

        # update
        fields.update_e()
        fields.update_h()

        # verify
        getf0, getf1 = {}, {}
        strfs_e = {'x':['ey', 'ez'], 'y':['ex', 'ez'], 'z':['ex', 'ey']}[axis]
        strfs_h = {'x':['hy', 'hz'], 'y':['hx', 'hz'], 'z':['hx', 'hy']}[axis]

        pt0 = (0, 0, 0)
        pt1 = { 'x': (0, ny-2, nz-2), \
                'y': (nx-2, 0, nz-2), \
                'z': (nx-2, ny-2, 0) }[axis]
        getf0['e'] = GetFields(fields, strfs_e, pt0, pt1)

        pt0 = { 'x': (nx-1, 0, 0), \
                'y': (0, ny-1, 0), \
                'z': (0, 0, nz-1) }[axis]
        pt1 = { 'x': (nx-1, ny-2, nz-2), \
                'y': (nx-2, ny-1, nz-2), \
                'z': (nx-2, ny-2, nz-1) }[axis]
        getf1['e'] = GetFields(fields, strfs_e, pt0, pt1)

        pt0 = { 'x': (0, 1, 1), \
                'y': (1, 0, 1), \
                'z': (1, 1, 0) }[axis]
        pt1 = { 'x': (0, ny-1, nz-1), \
                'y': (nx-1, 0, nz-1), \
                'z': (nx-1, ny-1, 0) }[axis]
        getf0['h'] = GetFields(fields, strfs_h, pt0, pt1)

        pt0 = { 'x': (nx-1, 1, 1), \
                'y': (1, ny-1, 1), \
                'z': (1, 1, nz-1) }[axis]
        pt1 = (nx-1, ny-1, nz-1)
        getf1['h'] = GetFields(fields, strfs_h, pt0, pt1)

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            norm = np.linalg.norm( \
                    getf0[eh].get_fields() - getf1[eh].get_fields() )
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, self.args, eh))



if __name__ == '__main__':
    nx, ny, nz = 40, 50, 60
    suite = unittest.TestSuite() 
    suite.addTest(TestPbc( ('x', nx, ny, nz, 'single') ))
    suite.addTest(TestPbc( ('z', nx, ny, 2, 'single') ))

    args_list = [ \
            (axis, nx, ny, nz, precision_float) \
            for axis in ['x', 'y', 'z'] \
            for nz in range(56, 73) \
            for precision_float in ['single', 'double'] ]
    suite.addTests(TestPbc(args) for args in args_list) 

    unittest.TextTestRunner().run(suite) 
