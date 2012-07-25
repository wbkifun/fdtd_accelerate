import numpy as np
import pyopencl as cl
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_gpu, common_update
from kemp.fdtd3d.node import NodeFields, NodeCore, NodePbc, NodeExchange
from kemp.fdtd3d import gpu, cpu


class TestNodePbcExchange(unittest.TestCase):
    def test_y_pbc_x_exchange(self):
        # instance
        nx, ny, nz = 40, 50, 60
        #nx, ny, nz = 3, 4, 5
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        gpuf = gpu.Fields(context, gpu_devices[0], nx, ny, nz)
        cpuf = cpu.Fields(nx, ny, nz)
        mainf_list = [gpuf, cpuf]
        nodef = NodeFields(mainf_list)
        core = NodeCore(nodef)
        pbc = NodePbc(nodef, 'y')
        exchange = NodeExchange(nodef)
        
        # generate random source
        ehs_gpu = common_update.generate_random_ehs(nx, ny, nz, nodef.dtype)
        gpuf.set_eh_bufs(*ehs_gpu)
        ehs_gpu_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], ehs_gpu) )

        ehs_cpu = common_update.generate_random_ehs(nx, ny, nz, nodef.dtype)
        cpuf.set_ehs(*ehs_cpu)
        ehs_cpu_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], ehs_cpu) )

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

        # x-axis exchange
        getf0['e'] = gpu.GetFields(gpuf, ['ey', 'ez'], (nx-1, 0, 0), (nx-1, ny-2, nz-2))
        getf1['e'] = cpu.GetFields(cpuf, ['ey', 'ez'], (0, 0, 0), (0, ny-2, nz-2))

        getf0['h'] = gpu.GetFields(gpuf, ['hy', 'hz'], (nx-1, 1, 1), (nx-1, ny-1, nz-1))
        getf1['h'] = cpu.GetFields(cpuf, ['hy', 'hz'], (0, 1, 1), (0, ny-1, nz-1))

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            g0 = getf0[eh].get_fields()
            g1 = getf1[eh].get_fields()
            norm = np.linalg.norm(g0 - g1)
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'x-axis exchange', eh))

        # y-axis pbc gpu
        getf0['e'] = gpu.GetFields(gpuf, ['ex', 'ez'], (0, ny-1, 0), (nx-2, ny-1, nz-2))
        getf1['e'] = gpu.GetFields(gpuf, ['ex', 'ez'], (0, 0, 0), (nx-2, 0, nz-2))

        getf0['h'] = gpu.GetFields(gpuf, ['hx', 'hz'], (1, ny-1, 1), (nx-1, ny-1, nz-1))
        getf1['h'] = gpu.GetFields(gpuf, ['hx', 'hz'], (1, 0, 1), (nx-1, 0, nz-1))

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            g0 = getf0[eh].get_fields()
            g1 = getf1[eh].get_fields()
            norm = np.linalg.norm(g0 - g1)
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'y-axis pbc gpu', eh))

        # y-axis pbc cpu
        getf0['e'] = cpu.GetFields(cpuf, ['ex', 'ez'], (0, ny-1, 0), (nx-2, ny-1, nz-2))
        getf1['e'] = cpu.GetFields(cpuf, ['ex', 'ez'], (0, 0, 0), (nx-2, 0, nz-2))

        getf0['h'] = cpu.GetFields(cpuf, ['hx', 'hz'], (1, ny-1, 1), (nx-1, ny-1, nz-1))
        getf1['h'] = cpu.GetFields(cpuf, ['hx', 'hz'], (1, 0, 1), (nx-1, 0, nz-1))

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            g0 = getf0[eh].get_fields()
            g1 = getf1[eh].get_fields()
            norm = np.linalg.norm(g0 - g1)
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'y-axis pbc cpu', eh))



    def test_z_pbc_x_exchange(self):
        # instance
        nx, ny, nz = 40, 50, 2
        #nx, ny, nz = 3, 4, 5
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)

        gpuf = gpu.Fields(context, gpu_devices[0], nx, ny, nz)
        cpuf = cpu.Fields(nx, ny, nz)
        mainf_list = [gpuf, cpuf]
        nodef = NodeFields(mainf_list)
        core = NodeCore(nodef)
        pbc = NodePbc(nodef, 'z')
        exchange = NodeExchange(nodef)
        
        # generate random source
        ehs_gpu = common_update.generate_random_ehs(nx, ny, nz, nodef.dtype)
        gpuf.set_eh_bufs(*ehs_gpu)
        ehs_gpu_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], ehs_gpu) )

        ehs_cpu = common_update.generate_random_ehs(nx, ny, nz, nodef.dtype)
        cpuf.set_ehs(*ehs_cpu)
        ehs_cpu_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], ehs_cpu) )

        # verify
        for tstep in xrange(1, 11):
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

        # x-axis exchange
        getf0['e'] = gpu.GetFields(gpuf, ['ey', 'ez'], (nx-1, 0, 0), (nx-1, ny-2, nz-2))
        getf1['e'] = cpu.GetFields(cpuf, ['ey', 'ez'], (0, 0, 0), (0, ny-2, nz-2))

        getf0['h'] = gpu.GetFields(gpuf, ['hy', 'hz'], (nx-1, 1, 1), (nx-1, ny-1, nz-1))
        getf1['h'] = cpu.GetFields(cpuf, ['hy', 'hz'], (0, 1, 1), (0, ny-1, nz-1))

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            g0 = getf0[eh].get_fields()
            g1 = getf1[eh].get_fields()
            norm = np.linalg.norm(g0 - g1)
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'x-axis exchange', eh))

        # z-axis pbc gpu
        getf0['e'] = gpu.GetFields(gpuf, ['ex', 'ey'], (0, 0, nz-1), (nx-2, ny-2, nz-1))
        getf1['e'] = gpu.GetFields(gpuf, ['ex', 'ey'], (0, 0, 0), (nx-2, ny-2, 0))

        getf0['h'] = gpu.GetFields(gpuf, ['hx', 'hy'], (1, 1, nz-1), (nx-1, ny-1, nz-1))
        getf1['h'] = gpu.GetFields(gpuf, ['hx', 'hy'], (1, 1, 0), (nx-1, ny-1, 0))

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            g0 = getf0[eh].get_fields()
            g1 = getf1[eh].get_fields()
            norm = np.linalg.norm(g0 - g1)
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'z-axis pbc gpu', eh))

        # z-axis pbc cpu
        getf0['e'] = cpu.GetFields(cpuf, ['ex', 'ey'], (0, 0,  nz-1), (nx-2, ny-2, nz-1))
        getf1['e'] = cpu.GetFields(cpuf, ['ex', 'ey'], (0, 0, 0), (nx-2, ny-2, 0))

        getf0['h'] = cpu.GetFields(cpuf, ['hx', 'hy'], (1, 1, nz-1), (nx-1, ny-1, nz-1))
        getf1['h'] = cpu.GetFields(cpuf, ['hx', 'hy'], (1, 1, 0), (nx-1, ny-1, 0))

        for getf in getf0.values() + getf1.values():
            getf.get_event().wait()

        for eh in ['e', 'h']:
            g0 = getf0[eh].get_fields()
            g1 = getf1[eh].get_fields()
            norm = np.linalg.norm(g0 - g1)
            self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'z-axis pbc cpu', eh))



if __name__ == '__main__':
    unittest.main()
