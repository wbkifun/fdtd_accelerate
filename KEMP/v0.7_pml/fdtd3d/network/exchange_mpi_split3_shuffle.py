import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.gpu import Fields, GetFields, SetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiSplit():
    def __init__(self, gpuf, core, direction, tmax):
        common.check_type('gpuf', gpuf, Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        if '+' in direction:
            self.gf_h = gf_h = GetFields(gpuf, ['hy', 'hz'], (-1, 0, 0), (-1, -1, -1)) 
            self.sf_e = SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True) 

            self.req_send_h = comm.Send_init(gf_h.host_array, rank+1, tag=0)
            self.tmp_recv_e_list = [np.zeros(gf_h.host_array.shape, gpuf.dtype) for i in range(2)]
            self.req_recv_e_list = [comm.Recv_init(tmp_recv_e, rank+1, tag=1) for tmp_recv_e in self.tmp_recv_e_list]
            self.switch_e = 0

        if '-' in direction:
            self.gf_e = gf_e = GetFields(gpuf, ['ey', 'ez'], (0, 0, 0), (0, -1, -1)) 
            self.sf_h = SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.req_send_e = comm.Send_init(gf_e.host_array, rank-1, tag=1)
            self.tmp_recv_h_list = [np.zeros(gf_e.host_array.shape, gpuf.dtype) for i in range(2)]
            self.req_recv_h_list = [comm.Recv_init(tmp_recv_h, rank-1, tag=0) for tmp_recv_h in self.tmp_recv_h_list]
            self.switch_h = 0

        # global variables
        self.gpuf = gpuf
        self.core = core
        self.direction = direction

        self.tmax = tmax
        self.tstep = 1



    def update_e(self):
        # update e
        self.core.update_e('pre')

        if '-' in self.direction:
            # mpi send e
            if self.tstep > 1: self.req_send_e.Wait()
            if self.tstep < self.tmax:
                self.gf_e.get_event().wait()
                self.req_send_e.Start()

        if '+' in self.direction:
            # mpi recv e
            if self.tstep > 1:
                self.req_recv_e_list[self.switch_e].Wait()
                self.sf_e.set_fields(self.tmp_recv_e_list[self.switch_e])
                self.switch_e = 1 if self.switch_e == 0 else 0
            if self.tstep < self.tmax: self.req_recv_e_list[self.switch_e].Start()

        # update h
        self.core.update_h('post')

        # update e
        self.core.update_e('mid')



    def update_h(self):
        # update h
        self.core.update_h('pre')

        if '+' in self.direction:
            # mpi send h
            if self.tstep > 1: self.req_send_h.Wait()
            if self.tstep < self.tmax: 
                self.gf_h.get_event().wait()
                self.req_send_h.Start()

        if '-' in self.direction:
            # mpi recv h
            if self.tstep > 1: 
                self.req_recv_h_list[self.switch_h].Wait()
                self.sf_h.set_fields(self.tmp_recv_h_list[self.switch_h])
                self.switch_h = 1 if self.switch_h == 0 else 0
            if self.tstep < self.tmax: self.req_recv_h_list[self.switch_h].Start()

        # update e
        self.core.update_e('post')

        # update h
        self.core.update_h('mid')

        self.tstep += 1
