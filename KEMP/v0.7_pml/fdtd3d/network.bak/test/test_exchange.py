import numpy as np
import unittest
from mpi4py import MPI
from time import sleep

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_random
from kemp.fdtd3d.network import ExchangeMpi
from kemp.fdtd3d import cpu, node


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class TestExchangeMpi_x(unittest.TestCase):
    def __init__(self, args):
        super(TestExchangeMpi_x, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz = self.args
        tmax = 10

        # instances
        buffer_dict = {}
        if rank == 0: buffer_dict['x+'] = cpu.BufferFields('x+', ny, nz, '', 'single')
        elif rank == 1: buffer_dict['x-'] = cpu.BufferFields('x-', ny, nz, '', 'single')

        import pyopencl as cl
        from kemp.fdtd3d.util import common_gpu
        from kemp.fdtd3d import gpu
        gpu_devices = common_gpu.gpu_device_list(print_info=False)
        context = cl.Context(gpu_devices)
        mainf_list = [ gpu.Fields(context, gpu_devices[0], nx, ny, nz) ]
        #mainf_list = [ cpu.Fields(nx, ny, nz, use_cpu_core=1) ]
        nodef = node.Fields(mainf_list, buffer_dict)

        # generate random source
        dtype = nodef.dtype
        ehs = common_random.generate_ehs(nx, ny, nz, dtype)
        buf_ehs = common_random.generate_ehs(3, ny, nz, dtype)
        #nodef.cpuf.set_ehs(*ehs)
        nodef.mainf_list[0].set_eh_bufs(*ehs)
        other = {0: 1, 1: 0}[rank]
        if rank == 0: 
            #nodef.buffer_dict['x+'].set_ehs(*buf_ehs)
            ExchangeMpi(nodef.buffer_dict['x+'], other, tmax)
        elif rank == 1:
            #nodef.buffer_dict['x-'].set_ehs(*buf_ehs)
            ExchangeMpi(nodef.buffer_dict['x-'], other, tmax)
        node.Core(nodef)

        # allocations for verify
        if rank == 0:
            getf_e = cpu.GetFields(nodef.buffer_dict['x+'], ['ey', 'ez'], (2, 0, 0), (2, ny-1, nz-1))
            getf_h = cpu.GetFields(nodef.buffer_dict['x+'], ['hy', 'hz'], (1, 0, 0), (1, ny-1, nz-1))
        elif rank == 1:
            getf_e = cpu.GetFields(nodef.buffer_dict['x-'], ['ey', 'ez'], (1, 0, 0), (1, ny-1, nz-1))
            getf_h = cpu.GetFields(nodef.buffer_dict['x-'], ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1))

        # verify
        print 'nodef, instance_list', rank, nodef.instance_list
        print 'f0, instance_list', rank, nodef.mainf_list[0].instance_list
        exch = nodef.instance_list[0]
        main_core = nodef.mainf_list[0].instance_list[0]
        if rank == 0: 
            #nodef.buffer_dict['x+'].instance_list.pop(0)
            print 'bufferf x+, instance_list', rank, nodef.buffer_dict['x+'].instance_list
            core, mpi = nodef.buffer_dict['x+'].instance_list
        elif rank == 1: 
            #nodef.buffer_dict['x-'].instance_list.pop(0)
            print 'bufferf x-, instance_list', rank, nodef.buffer_dict['x-'].instance_list
            core, mpi = nodef.buffer_dict['x-'].instance_list



        for tstep in xrange(1, tmax+1):
            #if rank == 0: print 'tstep', tstep

            #nodef.update_e()
            main_core.update_e()
            if rank == 0: 
                #print tstep, rank, 'core upE'
                core.update_e('')
                #print tstep, rank, 'mpi upE'
                mpi.update_e('')
            elif rank == 1: 
                #print tstep, rank, 'core upE pre'
                core.update_e('pre')
                #print tstep, rank, 'mpi upE pre'
                mpi.update_e('pre')
                #print tstep, rank, 'core upE post'
                core.update_e('post')
                #print tstep, rank, 'mpi upE post'
                mpi.update_e('post')
            exch.update_e()

            # verify the buffer
            #print tstep, rank, 'pre get'
            getf_h.get_event().wait()
            #print tstep, rank, 'after get'
            if rank == 1:
                #print tstep, rank, 'pre save'
                np.save('rank1_h_%d' % tstep, getf_h.get_fields())
                #print tstep, rank, 'after save'
            elif rank == 0:
                no_exist_npy = True
                while no_exist_npy:
                    try:
                        arr1 = np.load('rank1_h_%d.npy' % tstep)
                        no_exist_npy = False
                    except:
                        sleep(0.5)

                arr0 = getf_h.get_fields()
                #print tstep, 'h arr0\n', arr0
                #print tstep, 'h arr1\n', arr1
                norm = np.linalg.norm(arr0 - arr1)
                if norm != 0: print tstep, 'h norm', norm
                #if tstep > 1: self.assertEqual(norm, 0, '%s, %g, h' % (self.args, norm))


            #nodef.update_h()
            main_core.update_h()
            if rank == 0: 
                #print tstep, rank, 'core upH pre'
                core.update_h('pre')
                #print tstep, rank, 'mpi upH pre'
                mpi.update_h('pre')
                #print tstep, rank, 'core upH post'
                core.update_h('post')
                #print tstep, rank, 'mpi upH post'
                mpi.update_h('post')
            elif rank == 1: 
                #print tstep, rank, 'core upH'
                core.update_h('')
                #print tstep, rank, 'mpi upH'
                mpi.update_h('')
            exch.update_h()

            getf_e.get_event().wait()
            if rank == 1:
                np.save('rank1_e_%d' % tstep, getf_e.get_fields())
            elif rank == 0:
                no_exist_npy = True
                while no_exist_npy:
                    try:
                        arr1 = np.load('rank1_e_%d.npy' % tstep)
                        no_exist_npy = False
                    except:
                        sleep(0.5)

                arr0 = getf_e.get_fields()
                norm = np.linalg.norm(arr0 - arr1)
                if norm != 0: print tstep, 'e norm', norm
                #self.assertEqual(norm, 0, '%s, %g, e' % (self.args, norm))

        '''
        nodef.cpuf.enqueue_barrier()
        if rank == 0: nodef.buffer_dict['x+'].enqueue_barrier()
        if rank == 1: nodef.buffer_dict['x-'].enqueue_barrier()
        sleep(1) 
        '''




if __name__ == '__main__':
    #nx, ny, nz = 160, 140, 128
    nx, ny, nz = 500, 100, 128
    #nx, ny, nz = 7, 6, 4

    suite = unittest.TestSuite() 
    suite.addTest(TestExchangeMpi_x( (nx, ny, nz) ))

    unittest.TextTestRunner().run(suite) 
