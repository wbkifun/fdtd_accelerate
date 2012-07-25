import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.node import Fields
from kemp.fdtd3d.gpu import GetFields, SetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiNoBufferBlock:
    def __init__(self, node_fields, target_rank, direction):
        common.check_type('node_fields', node_fields, Fields)
        common.check_type('target_rank', target_rank, int)

        # local variables
        nodef = node_fields
        dtype = nodef.dtype

        assert rank != target_rank, 'The target_rank %d is same as the my_rank %d.' % (target_rank, rank)

        # create instances (getf, setf and mpi requests)
        if '+' in direction:
            mainf = nodef.mainf_list[-1]
            nx, ny, nz = mainf.ns
            getf = GetFields(mainf, ['hy', 'hz'], (nx-1, 0, 0), (nx-1, ny-1, nz-1))
            setf = SetFields(mainf, ['ey', 'ez'], (nx-1, 0, 0), (nx-1, ny-1, nz-1), True)

            if rank < target_rank:
                tag_send, tag_recv = 0, 1
            elif rank > target_rank:
                tag_send, tag_recv = 2, 3


        elif '-' in direction:
            mainf = nodef.mainf_list[0]
            nx, ny, nz = mainf.ns
            getf = GetFields(mainf, ['ey', 'ez'], (0, 0, 0), (0, ny-1, nz-1))
            setf = SetFields(mainf, ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1), True)

            if rank > target_rank:
                tag_send, tag_recv = 1, 0
            elif rank < target_rank:      # pbc
                tag_send, tag_recv = 3, 2


        # global variables
        self.target_rank = target_rank
        self.getf = getf
        self.setf = setf
        self.tag_send = tag_send
        self.tag_recv = tag_recv
        self.tmp_recv = np.zeros(getf.host_array.shape, dtype)

        # global functions
        self.update_e = self.recv if '+' in direction else self.send
        self.update_h = self.send if '+' in direction else self.recv

        # append instance
        self.priority_type = 'mpi'
        nodef.append_instance(self)


    def send(self):
        self.getf.get_event().wait()
        comm.Send(self.getf.host_array, self.target_rank, tag=self.tag_send)


    def recv(self):
        comm.Recv(self.tmp_recv, self.target_rank, tag=self.tag_recv)
        self.setf.set_fields(self.tmp_recv)
