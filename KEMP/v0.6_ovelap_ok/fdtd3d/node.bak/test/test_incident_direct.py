import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_random
from kemp.fdtd3d.node import IncidentDirect, Fields, GetFields
from kemp.fdtd3d import gpu, cpu


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
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        mainf_list = [ cpu.Fields(nx, ny, nz) ]
        mainf_list += [gpu.Fields(context, device, nx, ny, nz) \
                for device in gpu_devices]
        nodef = Fields(mainf_list)
        dtype = nodef.dtype
        anx = nodef.accum_nx_list

        tfunc = lambda tstep: np.sin(0.03*tstep)
        incident = IncidentDirect(nodef, str_f, pt0, pt1, tfunc, value) 

        # allocations for verify
        eh = np.zeros(nodef.ns, dtype)
        getf = GetFields(nodef, str_f, pt0, pt1)

        # verify
        eh[slices] = dtype(value) * dtype(tfunc(1))

        e_or_h = str_f[0]
        nodef.update_e()
        nodef.update_h()
        getf.wait()

        original = eh[slices]
        copy = getf.get_fields(str_f)
        norm = np.linalg.norm(original - copy)
        self.assertEqual(norm, 0, '%s, %g' % (self.args, norm))



if __name__ == '__main__':
    nx, ny, nz = 40, 50, 60
    gpu_devices = common_gpu.gpu_device_list(print_info=False)
    ngpu = len(gpu_devices)
    node_nx = (nx-1)*(ngpu+1) + 1

    suite = unittest.TestSuite() 
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), False) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), True) ))
    suite.addTest(TestIncidentDirect( (nx, ny, nz, 'ex', (0, 5, 12), (node_nx-1, 5, 12), True) ))

    # random sets
    args_list = [(nx, ny, nz, str_f, pt0, pt1, is_array) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in common_random.set_two_points(shape, node_nx, ny, nz) \
            for is_array in [False, True] ]
    test_size = int( len(args_list)*0.1 )
    test_list = [args_list.pop( np.random.randint(len(args_list)) ) for i in xrange(test_size)]
    suite.addTests(TestIncidentDirect(args) for args in test_list) 

    unittest.TextTestRunner().run(suite) 
