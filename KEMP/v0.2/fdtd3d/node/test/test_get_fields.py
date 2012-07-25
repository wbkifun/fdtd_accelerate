import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_update
from kemp.fdtd3d.util.common_test import random_set_two_points
from kemp.fdtd3d.node import NodeGetFields, NodeFields
from kemp.fdtd3d import gpu, cpu


class TestNodeGetFields(unittest.TestCase):
    def __init__(self, args):
        super(TestNodeGetFields, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1 = self.args

        slices = common.slice_index_two_points(pt0, pt1)
        str_fs = common.convert_to_tuple(str_f)

        # instance
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        mainf_list = [gpu.Fields(context, device, nx, ny, nz) \
                for device in gpu_devices]
        mainf_list.append( cpu.Fields(nx, ny, nz) )
        nodef = NodeFields(mainf_list)
        dtype = nodef.dtype
        anx = nodef.accum_nx_list

        getf = NodeGetFields(nodef, str_f, pt0, pt1) 
        
        # generate random source
        global_ehs = [np.zeros(nodef.ns, dtype) for i in range(6)]
        eh_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], global_ehs) )

        for i, f in enumerate(mainf_list[:-1]):
            nx, ny, nz = f.ns
            ehs = common_update.generate_random_ehs(nx, ny, nz, dtype)
            f.set_eh_bufs(*ehs)
            for eh, geh in zip(ehs, global_ehs):
                geh[anx[i]:anx[i+1],:,:] = eh[:-1,:,:]

        f = mainf_list[-1]
        nx, ny, nz = f.ns
        ehs = common_update.generate_random_ehs(nx, ny, nz, dtype)
        f.set_ehs(*ehs)
        for eh, geh in zip(ehs, global_ehs):
            geh[anx[-2]:anx[-1]+1,:,:] = eh[:]

        # verify
        getf.wait()

        for str_f in str_fs:
            original = eh_dict[str_f][slices]
            copy = getf.get_fields(str_f)
            norm = np.linalg.norm(original - copy)
            self.assertEqual(norm, 0, '%s, %g' % (self.args, norm))




if __name__ == '__main__':
    nx, ny, nz = 40, 50, 60
    gpu_devices = common_gpu.gpu_device_list(print_info=False)
    ngpu = len(gpu_devices)
    node_nx = (nx-1)*(ngpu+1) + 1

    suite = unittest.TestSuite() 
    suite.addTest(TestNodeGetFields( (nx, ny, nz, 'ex', (0, 0, 10), (node_nx-1, ny-1, 10)) ))
    suite.addTest(TestNodeGetFields( (nx, ny, nz, ['ex', 'ey'], (0, 1, 0), (0, 1, nz-1)) ))
    suite.addTest(TestNodeGetFields( (nx, ny, nz, ['ex'], (0, 1, 8), (0, 7, 16)) ))

    args_list = [(nx, ny, nz, str_f, pt0, pt1) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in random_set_two_points(shape, node_nx, ny, nz) ]
    test_size = int( len(args_list)*0.1 )
    test_list = [args_list.pop( np.random.randint(len(args_list)) ) for i in xrange(test_size)]
    suite.addTests(TestNodeGetFields(args) for args in test_list) 

    unittest.TextTestRunner().run(suite) 
