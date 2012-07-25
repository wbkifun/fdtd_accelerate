import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.gpu import Fields, GetFields, SetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiSplit2():
    def __init__(self, gpuf, core, direction):
        common.check_type('gpuf', gpuf, Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        if '+' in direction:
            self.gf_h = gf_h = GetFields(gpuf, ['hy', 'hz'], (-1, 0, 0), (-1, -1, -1)) 
            self.sf_e = SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True) 

            target = 0 if rank == size-1 else rank+1
            self.req_send_h = comm.Send_init(gf_h.host_array, target, tag=0)
            self.tmp_recv_e = np.zeros(gf_h.host_array.shape, gpuf.dtype)
            self.req_recv_e = comm.Recv_init(self.tmp_recv_e, target, tag=1)

        if '-' in direction:
            self.gf_e = gf_e = GetFields(gpuf, ['ey', 'ez'], (0, 0, 0), (0, -1, -1)) 
            self.sf_h = SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            target = size-1 if rank == 0 else rank-1
            self.req_send_e = comm.Send_init(gf_e.host_array, target, tag=1)
            self.tmp_recv_h = np.zeros(gf_e.host_array.shape, gpuf.dtype)
            self.req_recv_h = comm.Recv_init(self.tmp_recv_h, target, tag=0)

        # global variables
        self.core = core
        self.direction = direction



    def update_e(self):
        self.core.update_e('pre')

        if '-' in self.direction:
            self.gf_e.get_event().wait()
            self.req_send_e.Start()
        
        if '+' in self.direction:
            self.req_recv_e.Start()

        self.core.update_e('post')

        if '-' in self.direction:
            self.req_send_e.Wait()

        if '+' in self.direction:
            self.req_recv_e.Wait()
            self.sf_e.set_fields(self.tmp_recv_e)



    def update_h(self):
        self.core.update_h('pre')

        if '+' in self.direction:
            self.gf_h.get_event().wait()
            self.req_send_h.Start()
        
        if '-' in self.direction:
            self.req_recv_h.Start()

        self.core.update_h('post')

        if '+' in self.direction:
            self.req_send_h.Wait()

        if '-' in self.direction:
            self.req_recv_h.Wait()
            self.sf_h.set_fields(self.tmp_recv_h)
