import numpy as np
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_random
from kemp.fdtd3d.node import ExchangeNode
from kemp.fdtd3d import cpu, node


class TestExchangeNode_x(unittest.TestCase):
    def __init__(self, args):
        super(TestExchangeNode_x, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz = self.args

        # instances
        buffer_dict = {}
        buffer_dict['x+'] = cpu.BufferFields('x+', ny, nz, '', 'single')
        buffer_dict['x-'] = cpu.BufferFields('x-', ny, nz, '', 'single')

        import pyopencl as cl
        from kemp.fdtd3d.util import common_gpu
        from kemp.fdtd3d import gpu
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)
        mainf_list = [ gpu.Fields(context, gpu_devices[0], nx, ny, nz) ]
        #mainf_list = [ cpu.Fields(nx, ny, nz) ]
        nodef = node.Fields(mainf_list, buffer_dict)

        # generate random source
        dtype = nodef.dtype
        ehs = common_random.generate_ehs(nx, ny, nz, dtype)
        buf_ehs_p = common_random.generate_ehs(3, ny, nz, dtype)
        buf_ehs_m = common_random.generate_ehs(3, ny, nz, dtype)
        nodef.mainf_list[0].set_eh_bufs(*ehs)
        #nodef.mainf_list[0].set_ehs(*ehs)
        nodef.buffer_dict['x+'].set_ehs(*buf_ehs_p)
        nodef.buffer_dict['x-'].set_ehs(*buf_ehs_m)
        node.Core(nodef)

        # allocations for verify
        getf_dict = {'x+': {}, 'x-': {}}
        getf_buf_dict = {'x+': {}, 'x-': {}}

        getf_dict['x+']['e'] = gpu.GetFields(nodef.mainf_list[0], ['ey', 'ez'], (nx-1, 0, 0), (nx-1, ny-1, nz-1))
        getf_dict['x+']['h'] = gpu.GetFields(nodef.mainf_list[0], ['hy', 'hz'], (nx-2, 0, 0), (nx-2, ny-1, nz-1))

        getf_buf_dict['x+']['e'] = cpu.GetFields(nodef.buffer_dict['x+'], ['ey', 'ez'], (1, 0, 0), (1, ny-1, nz-1))
        getf_buf_dict['x+']['h'] = cpu.GetFields(nodef.buffer_dict['x+'], ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1))

        getf_dict['x-']['e'] = gpu.GetFields(nodef.mainf_list[0], ['ey', 'ez'], (1, 0, 0), (1, ny-1, nz-1))
        getf_dict['x-']['h'] = gpu.GetFields(nodef.mainf_list[0], ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1))

        getf_buf_dict['x-']['e'] = cpu.GetFields(nodef.buffer_dict['x-'], ['ey', 'ez'], (2, 0, 0), (2, ny-1, nz-1))
        getf_buf_dict['x-']['h'] = cpu.GetFields(nodef.buffer_dict['x-'], ['hy', 'hz'], (1, 0, 0), (1, ny-1, nz-1))

        # verify
        nodef.update_e()
        nodef.update_h()
        print 'nodef, instance_list', nodef.instance_list
        print 'mainf_list[0], instance_list', nodef.mainf_list[0].instance_list

        for direction in ['x+', 'x-']:
            for e_or_h in ['e', 'h']:
                getf = getf_dict[direction][e_or_h]
                getf_buf = getf_buf_dict[direction][e_or_h]

                getf.get_event().wait()
                getf_buf.get_event().wait()

                original = getf.get_fields()
                copy = getf_buf.get_fields()
                norm = np.linalg.norm(original - copy)
                self.assertEqual(norm, 0, '%s, %g, %s, %s' % (self.args, norm, direction, e_or_h))



if __name__ == '__main__':
    nx, ny, nz = 40, 50, 60

    suite = unittest.TestSuite() 
    suite.addTest(TestExchangeNode_x( (nx, ny, nz) ))

    unittest.TextTestRunner().run(suite) 
