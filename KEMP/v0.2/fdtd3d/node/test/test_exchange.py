import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_update
from kemp.fdtd3d.node import NodeFields, NodeExchange
from kemp.fdtd3d import gpu, cpu


class TestNodeExchange(unittest.TestCase):
    def __init__(self, args):
        super(TestNodeExchange, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz = self.args

        # instance
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        mainf_list = [gpu.Fields(context, device, nx, ny, nz) \
                for device in gpu_devices]
        mainf_list.append( cpu.Fields(nx, ny, nz) )
        nodef = NodeFields(mainf_list)
        dtype = nodef.dtype

        # buffer instance
        nodef.append_buffer_fields(cpu.Fields(3, ny, nz, mpi_type='x-'))
        nodef.append_buffer_fields(cpu.Fields(3, nodef.nx, nz, mpi_type='y+'))
        nodef.append_buffer_fields(cpu.Fields(3, nodef.nx, nz, mpi_type='y-'))
        nodef.append_buffer_fields(cpu.Fields(3, nodef.nx, ny, mpi_type='z+'))
        nodef.append_buffer_fields(cpu.Fields(3, nodef.nx, ny, mpi_type='z-'))

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
        exchange.update_e()
        exchange.update_h()
        getf0, getf1 = {}, {}

        # mainf list
        self.gpu, self.cpu = gpu, cpu
        for f0, f1 in zip(mainf_list[:-1], mainf_list[1:]):
            getf0['e'] = getattr(self, f0.device_type).GetFields(f0, ['ey', 'ez'], \
                    (f0.nx-1, 0, 0), (f0.nx-1, f0.ny-2, f0.nz-2))
            getf1['e'] = getattr(self, f1.device_type).GetFields(f1, ['ey', 'ez'], \
                    (0, 0, 0), (0, f1.ny-2, f1.nz-2))

            getf0['h'] = getattr(self, f0.device_type).GetFields(f0, ['hy', 'hz'], \
                    (f0.nx-1, 1, 1), (f0.nx-1, f0.ny-1, f0.nz-1))
            getf1['h'] = getattr(self, f1.device_type).GetFields(f1, ['hy', 'hz'], \
                    (0, 1, 1), (0, f1.ny-1, f1.nz-1))

            for getf in getf0.values() + getf1.values():
                getf.get_event().wait()

            for eh in ['e', 'h']:
                norm = np.linalg.norm(getf0[eh].get_fields() - getf1[eh].get_fields())
                self.assertEqual(norm, 0, '%s, %g, %s, %s, %s' % \
                        (self.args, norm, 'mainf', \
                        getf0[eh].mainf.device_type, getf1[eh].mainf.device_type) )

        # buffer 'x-'
        f0, f1 = nodef.cpuf_dict['x-'], mainf_list[0]
        getf0['e'] = cpu.GetFields(f0, ['ey', 'ez'], \
                (f0.nx-1, 0, 0), (f0.nx-1, f0.ny-2, f0.nz-2))
        getf1['e'] = gpu.GetFields(f1, ['ey', 'ez'], \
                (1, 0, 0), (1, f1.ny-2, f1.nz-2))

        getf0['h'] = cpu.GetFields(f0, ['hy', 'hz'], \
                (f0.nx-2, 1, 1), (f0.nx-2, f0.ny-1, f0.nz-1))
        getf1['h'] = gpu.GetFields(f1, ['hy', 'hz'], \
                (0, 1, 1), (0, f1.ny-1, f1.nz-1))

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            norm = np.linalg.norm( \
                    getf0[eh].get_fields() - getf1[eh].get_fields())
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'x-', eh) )

        # buffer 'y+'
        anx_list = nodef.accum_nx_list

        f1 = nodef.cpuf_dict['y+']
        for f0, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
            getf0['e'] = getattr(self, f0.device_type).GetFields(f0, ['ex', 'ez'], \
                    (0, f0.ny-1, 0), (f0.nx-2, f0.ny-1, f0.nz-2))
            getf1['e'] = cpu.GetFields(f1, ['ey', 'ez'], \
                    (1, anx0, 0), (1, anx1-1, f1.nz-2))

            getf0['h'] = getattr(self, f0.device_type).GetFields(f0, ['hx', 'hz'], \
                    (1, f0.ny-2, 1), (f0.nx-1, f0.ny-2, f0.nz-1))
            getf1['h'] = cpu.GetFields(f1, ['hy', 'hz'], \
                    (0, anx0+1, 1), (0, anx1, f1.nz-1))

            for getf in getf0.values() + getf1.values():
                getf.get_event().wait()

            for eh in ['e', 'h']:
                norm = np.linalg.norm( \
                        getf0[eh].get_fields() - getf1[eh].get_fields())
                self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'y+', eh) )

        # buffer 'y-'
        f0 = nodef.cpuf_dict['y-']
        for f1, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
            getf0['e'] = cpu.GetFields(f0, ['ey', 'ez'], \
                    (f0.nx-1, anx0, 0), (f0.nx-1, anx1-1, f0.nz-2))
            getf1['e'] = getattr(self, f1.device_type).GetFields(f1, ['ex', 'ez'], \
                    (0, 1, 0), (f1.nx-2, 1, f1.nz-2))

            getf0['h'] = cpu.GetFields(f0, ['hy', 'hz'], \
                    (f0.nx-2, anx0+1, 1), (f0.nx-2, anx1, f0.nz-1))
            getf1['h'] = getattr(self, f1.device_type).GetFields(f1, ['hx', 'hz'], \
                    (1, 0, 1), (f1.nx-1, 0, f1.nz-1))

            for getf in getf0.values() + getf1.values():
                getf.get_event().wait()

            for eh in ['e', 'h']:
                norm = np.linalg.norm( \
                        getf0[eh].get_fields() - getf1[eh].get_fields())
                self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'y-', eh) )

        # buffer 'z+'
        f1 = nodef.cpuf_dict['z+']
        for f0, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
            getf0['e'] = getattr(self, f0.device_type).GetFields(f0, ['ex', 'ey'], \
                    (0, 0, f0.nz-1), (f0.nx-2, f0.ny-2, f0.nz-1))
            getf1['e'] = cpu.GetFields(f1, ['ey', 'ez'], \
                    (1, anx0, 0), (1, anx1-1, f1.nz-2))

            getf0['h'] = getattr(self, f0.device_type).GetFields(f0, ['hx', 'hy'], \
                    (1, 1, f0.nz-2), (f0.nx-1, f0.ny-1, f0.nz-2))
            getf1['h'] = cpu.GetFields(f1, ['hy', 'hz'], \
                    (0, anx0+1, 1), (0, anx1, f1.nz-1))

            for getf in getf0.values() + getf1.values():
                getf.get_event().wait()

            for eh in ['e', 'h']:
                norm = np.linalg.norm( \
                        getf0[eh].get_fields() - getf1[eh].get_fields())
                self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'z+', eh) )

        # buffer 'z-'
        f0 = nodef.cpuf_dict['z-']
        for f1, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
            getf0['e'] = cpu.GetFields(f0, ['ey', 'ez'], \
                    (f0.nx-1, anx0, 0), (f0.nx-1, anx1-1, f0.nz-2))
            getf1['e'] = getattr(self, f1.device_type).GetFields(f1, ['ex', 'ey'], \
                    (0, 0, 1), (f1.nx-2, f1.ny-2, 1))

            getf0['h'] = cpu.GetFields(f0, ['hy', 'hz'], \
                    (f0.nx-2, anx0+1, 1), (f0.nx-2, anx1, f0.nz-1))
            getf1['h'] = getattr(self, f1.device_type).GetFields(f1, ['hx', 'hy'], \
                    (1, 1, 0), (f1.nx-1, f1.ny-1, 0))

            for getf in getf0.values() + getf1.values():
                getf.get_event().wait()

            for eh in ['e', 'h']:
                norm = np.linalg.norm( \
                        getf0[eh].get_fields() - getf1[eh].get_fields())
                self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'z-', eh) )



if __name__ == '__main__':
    nx, ny, nz = 40, 50, 60
    suite = unittest.TestSuite() 

    suite.addTest(TestNodeExchange( (nx, ny, nz) ))

    unittest.TextTestRunner().run(suite) 
