import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiNonBlock():
    def __init__(self, gpuf, direction):
        common.check_type('gpuf', gpuf, gpu.Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        if '+' in direction:
            self.getf_h = getf_h = gpu.GetFields(gpuf, ['hy', 'hz'], (-1, 0, 0), (-1, -1, -1)) 
            self.setf_e = gpu.SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.req_send_h = comm.Send_init(getf_h.host_array, rank+1, tag=0)
            self.tmp_recv_e = tmp_recv_e = np.zeros(getf_h.host_array.shape, gpuf.dtype)
            self.req_recv_e = comm.Recv_init(tmp_recv_e, rank+1, tag=1)

        if '-' in direction:
            self.getf_e = getf_e = gpu.GetFields(gpuf, ['ey', 'ez'], (0, 0, 0), (0, -1, -1)) 
            self.setf_h = gpu.SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.req_send_e = comm.Send_init(getf_e.host_array, rank-1, tag=1)
            self.tmp_recv_h = tmp_recv_h = np.zeros(getf_e.host_array.shape, gpuf.dtype)
            self.req_recv_h = comm.Recv_init(tmp_recv_h, rank-1, tag=0)

        # global variables
        self.direction = direction


    def update_e(self):
        if '+' in self.direction:
            self.req_recv_e.Start()

        if '-' in self.direction:
            self.getf_e.get_event().wait()
            self.req_send_e.Start()
        

        if '+' in self.direction:
            self.req_recv_e.Wait()
            self.setf_e.set_fields(self.tmp_recv_e)

        if '-' in self.direction:
            self.req_send_e.Wait()


    def update_h(self):
        if '-' in self.direction:
            self.req_recv_h.Start()

        if '+' in self.direction:
            self.getf_h.get_event().wait()
            self.req_send_h.Start()
        

        if '-' in self.direction:
            self.req_recv_h.Wait()
            self.setf_h.set_fields(self.tmp_recv_h)

        if '+' in self.direction:
            self.req_send_h.Wait()
