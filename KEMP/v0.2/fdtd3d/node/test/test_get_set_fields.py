import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_random
from kemp.fdtd3d.node import GetFields, SetFields, Fields
from kemp.fdtd3d import gpu, cpu


class TestGetFields(unittest.TestCase):
    def __init__(self, args):
        super(TestGetFields, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1 = self.args

        slices = common.slices_two_points(pt0, pt1)
        str_fs = common.convert_to_tuple(str_f)

        # instance
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        mainf_list = [gpu.Fields(context, device, nx, ny, nz) \
                for device in gpu_devices]
        mainf_list.append( cpu.Fields(nx, ny, nz) )
        nodef = Fields(mainf_list)
        dtype = nodef.dtype
        anx = nodef.accum_nx_list

        getf = GetFields(nodef, str_f, pt0, pt1) 
        
        # generate random source
        global_ehs = [np.zeros(nodef.ns, dtype) for i in range(6)]
        eh_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], global_ehs) )

        for i, f in enumerate(mainf_list[:-1]):
            nx, ny, nz = f.ns
            ehs = common_random.generate_ehs(nx, ny, nz, dtype)
            f.set_eh_bufs(*ehs)
            for eh, geh in zip(ehs, global_ehs):
                geh[anx[i]:anx[i+1],:,:] = eh[:-1,:,:]

        f = mainf_list[-1]
        nx, ny, nz = f.ns
        ehs = common_random.generate_ehs(nx, ny, nz, dtype)
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



class TestSetFields(unittest.TestCase):
    def __init__(self, args):
        super(TestSetFields, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1, is_array = self.args

        slices = common.slices_two_points(pt0, pt1)
        str_fs = common.convert_to_tuple(str_f)

        # instance
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        mainf_list = [gpu.Fields(context, device, nx, ny, nz) \
                for device in gpu_devices]
        mainf_list.append( cpu.Fields(nx, ny, nz) )
        nodef = Fields(mainf_list)
        dtype = nodef.dtype
        anx = nodef.accum_nx_list

        getf = GetFields(nodef, str_f, (0, 0, 0), (nodef.nx-1, ny-1, nz-1)) 
        setf = SetFields(nodef, str_f, pt0, pt1, is_array) 
        
        # generate random source
        if is_array:
            shape = common.shape_two_points(pt0, pt1, len(str_fs))
            value = np.random.rand(*shape).astype(nodef.dtype)
            split_value = np.split(value, len(str_fs))
            split_value_dict = dict( zip(str_fs, split_value) )
        else:
            value = np.random.ranf()

        # host allocations
        global_ehs = [np.zeros(nodef.ns, dtype) for i in range(6)]
        eh_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], global_ehs) )

        # verify
        for str_f in str_fs:
            if is_array:
                eh_dict[str_f][slices] = split_value_dict[str_f]
            else:
                eh_dict[str_f][slices] = value

        setf.set_fields(value)
        gpu_getf = gpu.GetFields(mainf_list[0], str_fs, (0, 0, 0), (nx-1, ny-1, nz-1))
        gpu_getf.get_event().wait()
        getf.wait()
        

        for str_f in str_fs:
            original = eh_dict[str_f]
            copy = getf.get_fields(str_f)
            norm = np.linalg.norm(original - copy)
            #if norm != 0:
                #print '\ngpu getf\n', gpu_getf.get_fields(str_f)
                #print original[slices]
                #print copy[slices]
            self.assertEqual(norm, 0, '%s, %g, %s' % (self.args, norm, str_f))




if __name__ == '__main__':
    nx, ny, nz = 4, 5, 6
    gpu_devices = common_gpu.gpu_device_list(print_info=False)
    ngpu = len(gpu_devices)
    node_nx = (nx-1)*(ngpu+1) + 1

    suite = unittest.TestSuite() 

    # Test GetFields
    '''
    suite.addTest(TestGetFields( (nx, ny, nz, 'ex', (0, 0, 10), (node_nx-1, ny-1, 10)) ))
    suite.addTest(TestGetFields( (nx, ny, nz, ['ex', 'ey'], (0, 1, 0), (0, 1, nz-1)) ))
    suite.addTest(TestGetFields( (nx, ny, nz, ['ex'], (0, 1, 8), (0, 7, 16)) ))

    args_list = [(nx, ny, nz, str_f, pt0, pt1) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in common_random.set_two_points(shape, node_nx, ny, nz) ]
    test_size = int( len(args_list)*0.1 )
    test_list = [args_list.pop( np.random.randint(len(args_list)) ) for i in xrange(test_size)]
    suite.addTests(TestGetFields(args) for args in test_list) 
    '''

    # Test SetFields
    suite.addTest(TestSetFields( (nx, ny, nz, 'ex', (0, 1, 0), (node_nx-1, 1, nz-1), False) ))
    suite.addTest(TestSetFields( (nx, ny, nz, ['ex', 'ey'], (0, 1, 0), (node_nx-1, 1, nz-1), False) ))
    suite.addTest(TestSetFields( (nx, ny, nz, 'ex', (0, 1, 0), (node_nx-1, 1, nz-1), True) ))
    suite.addTest(TestSetFields( (nx, ny, nz, ['ex', 'ey'], (0, 1, 0), (node_nx-1, 1, nz-1), True) ))

    args_list2 = [(nx, ny, nz, str_f, pt0, pt1, is_array) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in common_random.set_two_points(shape, node_nx, ny, nz) \
            for is_array in [False, True] ]
    test_size = int( len(args_list2)*0.1 )
    test_list = [args_list2.pop( np.random.randint(len(args_list2)) ) for i in xrange(test_size)]
    suite.addTests(TestSetFields(args) for args in test_list) 

    unittest.TextTestRunner().run(suite) 
