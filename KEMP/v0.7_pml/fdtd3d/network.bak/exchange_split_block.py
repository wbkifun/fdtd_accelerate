import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.cpu import BufferFields, GetFields, SetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiSplitBlock:
    def __init__(self, buffer_fields):
        common.check_type('buffer_fields', buffer_fields, BufferFields)

        # local variables
        mainf = buffer_fields
        nx, ny, nz = mainf.ns
        dtype = mainf.dtype
        direction = mainf.direction
        target_rank = mainf.target_rank

        # create instances (getf, setf and mpi requests)
        if '+' in direction:     # split h
            getf = GetFields(mainf, ['hy', 'hz'], (1, 0, 0), (1, ny-1, nz-1))
            setf = SetFields(mainf, ['ey', 'ez'], (2, 0, 0), (2, ny-1, nz-1), True)

            if rank < target_rank:
                tag_send, tag_recv = 0, 1
            elif rank > target_rank:    # pbc
                tag_send, tag_recv = 2, 3

        elif '-' in direction:   # split e
            getf = GetFields(mainf, ['ey', 'ez'], (1, 0, 0), (1, ny-1, nz-1))
            setf = SetFields(mainf, ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1), True)

            if rank > target_rank:
                tag_send, tag_recv = 1, 0
            elif rank < target_rank:      # pbc
                tag_send, tag_recv = 3, 2

        # global variables and functions
        self.target_rank = target_rank
        self.getf = getf
        self.setf = setf
        self.tag_send = tag_send
        self.tag_recv = tag_recv
        self.tmp_recv = np.zeros(getf.host_array.shape, dtype)

        # global functions
        self.update_e = self.recv if '+' in direction else self.send
        self.update_h = self.send if '+' in direction else self.recv

        # append to the update list
        self.priority_type = 'mpi'
        mainf.append_instance(self)


    def send(self, part):
        if part in ['', 'post']:
            self.getf.get_event().wait()
            comm.Send(self.getf.host_array, self.target_rank, self.tag_send)


    def recv(self, part):
        if part in ['', 'post']:
            comm.Recv(self.tmp_recv, self.target_rank, self.tag_recv)
            self.setf.set_fields(self.tmp_recv)
