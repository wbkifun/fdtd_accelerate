import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.gpu import Fields, GetFields, SetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiSplit3():
    def __init__(self, gpuf, core, direction, tmax):
        common.check_type('gpuf', gpuf, Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        if '+' in direction:
            self.gf_h = gf_h = GetFields(gpuf, ['hy', 'hz'], (-2, 0, 0), (-2, -1, -1)) 
            self.sf_e = SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True) 

            self.req_send_h = comm.Send_init(gf_h.host_array, rank+1, tag=0)
            self.tmp_recv_e_list = [np.zeros(gf_h.host_array.shape, gpuf.dtype) for i in range(2)]
            self.req_recv_e_list = [comm.Recv_init(tmp_recv_e, rank+1, tag=1) for tmp_recv_e in self.tmp_recv_e_list]
            self.switch_e = 0

        if '-' in direction:
            self.gf_e = gf_e = GetFields(gpuf, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_h = SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.req_send_e = comm.Send_init(gf_e.host_array, rank-1, tag=1)
            self.tmp_recv_h_list = [np.zeros(gf_e.host_array.shape, gpuf.dtype) for i in range(2)]
            self.req_recv_h_list = [comm.Recv_init(tmp_recv_h, rank-1, tag=0) for tmp_recv_h in self.tmp_recv_h_list]
            self.switch_h = 0

        # global variables
        self.core = core
        self.direction = direction

        self.tmax = tmax
        self.tstep = 1



    def update_e(self):
        if self.direction == '+':
            self.core.update_e()

        else:
            # update e
            self.core.update_e('pre')

            # mpi send e
            if self.tstep > 1: self.req_send_e.Wait()
            self.gf_e.get_event().wait()
            self.req_send_e.Start()
            if self.tstep == self.tmax: self.req_send_e.Wait()

            # update e
            self.core.update_e('mid')

            # mpi recv h
            if self.tstep > 1: 
                self.req_recv_h_list[self.switch_h].Wait()
                self.sf_h.set_fields(self.tmp_recv_h_list[self.switch_h])
                self.switch_h = 1 if self.switch_h == 0 else 0
            if self.tstep < self.tmax: self.req_recv_h_list[self.switch_h].Start()

            # update e
            self.core.update_e('post')


    def update_h(self):
        if self.direction == '-':
            self.core.update_h()

        else:
            # update h
            self.core.update_h('pre')

            # mpi send h
            if self.tstep > 1: self.req_send_h.Wait()
            if self.tstep < self.tmax: 
                self.gf_h.get_event().wait()
                self.req_send_h.Start()

            # update h
            self.core.update_h('mid')

            # mpi recv e
            if self.tstep == 1: self.req_recv_e_list[self.switch_e].Start()
            self.req_recv_e_list[self.switch_e].Wait()
            self.sf_e.set_fields(self.tmp_recv_e_list[self.switch_e])
            self.switch_e = 1 if self.switch_e == 0 else 0
            if self.tstep < self.tmax: self.req_recv_e_list[self.switch_e].Start()

            # update h
            self.core.update_h('post')

        self.tstep += 1
