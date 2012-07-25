import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_update
from kemp.fdtd3d.node import NodeFields, NodePbc, NodeExchange
from kemp.fdtd3d import gpu, cpu


class TestNodePbc(unittest.TestCase):
    def __init__(self, args):
        super(TestNodePbc, self).__init__() 
        self.args = args


    def runTest(self):
        axis, nx, ny, nz = self.args
        self.gpu, self.cpu = gpu, cpu

        # instance
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        mainf_list = [gpu.Fields(context, device, nx, ny, nz) \
                for device in gpu_devices]
        mainf_list.append( cpu.Fields(nx, ny, nz) )
        nodef = NodeFields(mainf_list)
        dtype = nodef.dtype

        pbc = NodePbc(nodef, axis)
        exchange = NodeExchange(nodef)
        
        # generate random source
        for f in mainf_list[:-1]:
            nx, ny, nz = f.ns
            ehs = common_update.generate_random_ehs(nx, ny, nz, dtype)
            f.set_eh_bufs(*ehs)

        for f in nodef.cpuf_dict.values():
            nx, ny, nz = f.ns
            ehs = common_update.generate_random_ehs(nx, ny, nz, dtype)
            f.set_ehs(*ehs)

        # verify
        for mainf in mainf_list:
            mainf.update_e()
        pbc.update_e()
        exchange.update_e()

        for mainf in mainf_list:
            mainf.update_h()
        pbc.update_h()
        exchange.update_h()

        mainf_list[-1].enqueue_barrier()

        getf0, getf1 = {}, {}

        if axis == 'x':
            f0, f1 = mainf_list[0], mainf_list[-1]
            getf0['e'] = getattr(self, f0.device_type).GetFields(f0, ['ey', 'ez'], \
                    (0, 0, 0), (0, f0.ny-2, f0.nz-2))
            getf1['e'] = getattr(self, f1.device_type).GetFields(f1, ['ey', 'ez'], \
                    (f1.nx-1, 0, 0), (f1.nx-1, f1.ny-2, f1.nz-2))

            getf0['h'] = getattr(self, f0.device_type).GetFields(f0, ['hy', 'hz'], \
                    (0, 1, 1), (0, f0.ny-1, f0.nz-1))
            getf1['h'] = getattr(self, f1.device_type).GetFields(f1, ['hy', 'hz'], \
                    (f1.nx-1, 1, 1), (f1.nx-1, f1.ny-1, f1.nz-1))

            for getf in getf0.values() + getf1.values():
                getf.get_event().wait()

            for eh in ['e', 'h']:
                norm = np.linalg.norm(getf0[eh].get_fields() - getf1[eh].get_fields())
                self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'x', eh))

        elif axis == 'y':
            for f in mainf_list:
                getf0['e'] = getattr(self, f.device_type).GetFields(f, ['ex', 'ez'], \
                        (0, 0, 0), (f.nx-2, 0, f.nz-2))
                getf1['e'] = getattr(self, f.device_type).GetFields(f, ['ex', 'ez'], \
                        (0, f.ny-1, 0), (f.nx-2, f.ny-1, f.nz-2))

                getf0['h'] = getattr(self, f.device_type).GetFields(f, ['hx', 'hz'], \
                        (1, 0, 1), (f.nx-1, 0, f.nz-1))
                getf1['h'] = getattr(self, f.device_type).GetFields(f, ['hx', 'hz'], \
                        (1, f.ny-1, 1), (f.nx-1, f.ny-1, f.nz-1))

                for getf in getf0.values() + getf1.values():
                    getf.get_event().wait()

                for eh in ['e', 'h']:
                    norm = np.linalg.norm( \
                            getf0[eh].get_fields() - getf1[eh].get_fields())
                    self.assertEqual(norm, 0, '%g, %s, %s, %s' % (norm, 'y', eh, f.device_type) )

        elif axis == 'z':
            for f in mainf_list:
                getf0['e'] = getattr(self, f.device_type).GetFields(f, ['ex', 'ey'], \
                        (0, 0, f.nz-1), (f.nx-2, f.ny-2, f.nz-1))
                getf1['e'] = getattr(self, f.device_type).GetFields(f, ['ex', 'ey'], \
                        (0, 0, 0), (f.nx-2, f.ny-2, 0))

                getf0['h'] = getattr(self, f.device_type).GetFields(f, ['hx', 'hy'], \
                        (1, 1, f.nz-1), (f.nx-1, f.ny-1, f.nz-1))
                getf1['h'] = getattr(self, f.device_type).GetFields(f, ['hx', 'hy'], \
                        (1, 1, 0), (f.nx-1, f.ny-1, 0))

                for getf in getf0.values() + getf1.values():
                    getf.get_event().wait()

                for eh in ['e', 'h']:
                    norm = np.linalg.norm( \
                            getf0[eh].get_fields() - getf1[eh].get_fields())
                    self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'z', eh) )



if __name__ == '__main__':
    nx, ny, nz = 40, 50, 60
    suite = unittest.TestSuite() 
    suite.addTest( TestNodePbc(('x', nx, ny, nz)) )

    args_list = [ \
            (axis, nx, ny, nz) \
            for axis in ['x', 'y', 'z'] ]
    suite.addTests(TestNodePbc(args) for args in args_list) 

    unittest.TextTestRunner().run(suite) 
