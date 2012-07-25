import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d import node, gpu, cpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiBuffer():
    def __init__(self, nodef):
        common.check_type('nodef', nodef, node.Fields)

        self.directions = nodef.buffer_dict.keys()
        self.gpu = gpu
        self.cpu = cpu

        if 'x+' in self.directions:
            mf_xp = nodef.mainf_list[-1]
            bf_xp = nodef.buffer_dict['x+']
            mpu = getattr(self, mf_xp.device_type)

            self.gf_p_h = mpu.GetFields(mf_xp, ['hy', 'hz'], (-2, 0, 0), (-2, -1, -1)) 
            self.sf_p_h = cpu.SetFields(bf_xp, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.sf_p_e = mpu.SetFields(mf_xp, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.gf_h = cpu.GetFields(bf_xp, ['hy', 'hz'], (1, 0, 0), (1, -1, -1)) 
            self.sf_e = cpu.SetFields(bf_xp, ['ey', 'ez'], (1, 0, 0), (1, -1, -1), True) 

            target = 0 if rank == size-1 else rank+1
            self.req_send_h = comm.Send_init(self.gf_h.host_array, target, tag=0)
            self.tmp_recv_e = np.zeros(self.gf_h.host_array.shape, nodef.dtype)
            self.req_recv_e = comm.Recv_init(self.tmp_recv_e, target, tag=1)

        if 'x-' in self.directions:
            mf_xm = nodef.mainf_list[0]
            bf_xm = nodef.buffer_dict['x-']
            mpu = getattr(self, mf_xm.device_type)

            self.gf_m_e = mpu.GetFields(mf_xm, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_m_e = cpu.SetFields(bf_xm, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.sf_m_h = mpu.SetFields(mf_xm, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.gf_e = cpu.GetFields(bf_xm, ['ey', 'ez'], (0, 0, 0), (0, -1, -1)) 
            self.sf_h = cpu.SetFields(bf_xm, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True) 

            target = size-1 if rank == 0 else rank-1
            self.req_send_e = comm.Send_init(self.gf_e.host_array, target, tag=1)
            self.tmp_recv_h = np.zeros(self.gf_e.host_array.shape, nodef.dtype)
            self.req_recv_h = comm.Recv_init(self.tmp_recv_h, target, tag=0)

        # append to the update list
        self.priority_type = 'mpi'
        nodef.append_instance(self)



    def update_e(self):
        if 'x-' in self.directions:
            self.gf_e.get_event().wait()
            self.req_send_e.Start()
        
        if 'x+' in self.directions:
            self.req_recv_e.Start()

        if 'x-' in self.directions:
            self.req_send_e.Wait()

        if 'x+' in self.directions:
            self.req_recv_e.Wait()
            self.sf_e.set_fields(self.tmp_recv_e)
            self.sf_p_e.set_fields(self.tmp_recv_e)

        if 'x-' in self.directions:
            self.sf_m_e.set_fields(self.gf_m_e.get_fields(), [self.gf_m_e.get_event()])


    def update_h(self):
        if 'x+' in self.directions:
            self.gf_h.get_event().wait()
            self.req_send_h.Start()
        
        if 'x-' in self.directions:
            self.req_recv_h.Start()

        if 'x+' in self.directions:
            self.req_send_h.Wait()

        if 'x-' in self.directions:
            self.req_recv_h.Wait()
            self.sf_h.set_fields(self.tmp_recv_h)
            self.sf_m_h.set_fields(self.tmp_recv_h)

        if 'x+' in self.directions:
            self.sf_p_h.set_fields(self.gf_p_h.get_fields(), [self.gf_p_h.get_event()])
