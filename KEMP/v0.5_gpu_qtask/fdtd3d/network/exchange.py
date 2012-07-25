import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.cpu import BufferFields, GetFields, SetFields

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpi:
    def __init__(self, buffer_fields, target_rank, max_tstep):
        common.check_type('buffer_fields', buffer_fields, BufferFields)
        common.check_type('target_rank', target_rank, int)

        # local variables
        mainf = buffer_fields
        nx, ny, nz = mainf.ns
        dtype = mainf.dtype
        direction = mainf.direction

        assert rank != target_rank, 'The target_rank %d is same as the my_rank %d.' % (target_rank, rank)

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

        req_send = comm.Send_init(getf.host_array, target_rank, tag=tag_send)
        tmp_recv = np.zeros(getf.host_array.shape, dtype) 
        req_recv = comm.Recv_init(tmp_recv, target_rank, tag=tag_recv)

        # global variables and functions
        self.mainf = mainf
        self.getf = getf
        self.setf = setf
        self.req_send = req_send
        self.tmp_recv = tmp_recv
        self.req_recv = req_recv

        self.max_tstep = max_tstep
        self.tstep = 1

        # append to the update list
        self.priority_type = 'mpi'
        mainf.append_instance(self)


    def update_e(self, part):
        if part == '' and self.tstep == 1:
            self.mainf.enqueue(self.req_recv.Start)

        elif part == 'pre':
            # send
            if self.tstep > 1:
                self.mainf.enqueue(self.req_send.Wait)

            evt = self.getf.get_event()
            self.mainf.enqueue(self.req_send.Start, wait_for=[evt])

            # recv
            if self.tstep > 1:
                self.mainf.enqueue(self.req_recv.Wait)
                self.setf.set_fields(self.tmp_recv)

            if self.tstep < self.max_tstep:
                self.mainf.enqueue(self.req_recv.Start)


    def update_h(self, part):
        if part == '' and self.tstep == self.max_tstep:
            self.mainf.enqueue(self.req_send.Wait)

        elif part == 'pre':
            # send
            if self.tstep > 1:
                self.mainf.enqueue(self.req_send.Wait)

            if self.tstep < self.max_tstep:
                evt = self.getf.get_event()
                self.mainf.enqueue(self.req_send.Start, wait_for=[evt])

            # recv
            self.mainf.enqueue(self.req_recv.Wait)
            self.setf.set_fields(self.tmp_recv)

            if self.tstep < self.max_tstep:
                self.mainf.enqueue(self.req_recv.Start)

        if part in ['', 'pre']:
            self.tstep += 1
