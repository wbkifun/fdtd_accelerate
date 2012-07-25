import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.node import Fields
from kemp.fdtd3d.gpu import GetFields, SetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiNoBufferNonBlock:
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
                tag0, tag1 = 0, 1
            elif rank > target_rank:
                tag0, tag1 = 2, 3

        elif '-' in direction:   # split e
            mainf = nodef.mainf_list[0]
            nx, ny, nz = mainf.ns
            getf = GetFields(mainf, ['ey', 'ez'], (0, 0, 0), (0, ny-1, nz-1))
            setf = SetFields(mainf, ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1), True)

            if rank > target_rank:
                tag0, tag1 = 1, 0
            elif rank < target_rank:      # pbc
                tag0, tag1 = 3, 2

        req_send = comm.Send_init(getf.host_array, target_rank, tag=tag0)
        tmp_recv = np.zeros(getf.host_array.shape, dtype)
        req_recv = comm.Recv_init(tmp_recv, target_rank, tag=tag1)

        # global variables
        self.target_rank = target_rank
        self.getf = getf
        self.setf = setf
        self.req_send = req_send
        self.req_recv = req_recv
        self.tmp_recv = tmp_recv

        # global functions
        self.update_e = self.recv if '+' in direction else self.send
        self.update_h = self.send if '+' in direction else self.recv


    def send(self, part):
        if part == 'pre':
            self.getf.get_event().wait()
            self.req_send.Start()
        elif part == 'post':
            self.req_send.Wait()


    def recv(self, part):
        if part == 'pre':
            self.req_recv.Start()
        elif part == 'post':
            self.req_recv.Wait()
            self.setf.set_fields(self.tmp_recv)
