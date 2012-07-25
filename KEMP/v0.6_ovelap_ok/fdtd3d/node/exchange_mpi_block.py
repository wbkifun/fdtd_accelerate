import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiBlock():
    def __init__(self, gpuf, direction):
        common.check_type('gpuf', gpuf, gpu.Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        if '+' in direction:
            self.getf_h = getf_h = gpu.GetFields(gpuf, ['hy', 'hz'], (-1, 0, 0), (-1, -1, -1)) 
            self.setf_e = gpu.SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.tmp_recv_e = tmp_recv_e = np.zeros(getf_h.host_array.shape, gpuf.dtype)

        if '-' in direction:
            self.getf_e = getf_e = gpu.GetFields(gpuf, ['ey', 'ez'], (0, 0, 0), (0, -1, -1)) 
            self.setf_h = gpu.SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.tmp_recv_h = tmp_recv_h = np.zeros(getf_e.host_array.shape, gpuf.dtype)

        # global variables
        self.direction = direction


    def update_e(self):
        if '-' in self.direction:
            self.getf_e.get_event().wait()
            comm.Send(self.getf_e.get_fields(), rank-1, 1)

        if '+' in self.direction:
            comm.Recv(self.tmp_recv_e, rank+1, 1)
            self.setf_e.set_fields(self.tmp_recv_e)


    def update_h(self):
        if '+' in self.direction:
            self.getf_h.get_event().wait()
            comm.Send(self.getf_h.get_fields(), rank+1, 0)

        if '-' in self.direction:
            comm.Recv(self.tmp_recv_h, rank-1, 0)
            self.setf_h.set_fields(self.tmp_recv_h)
