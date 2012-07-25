import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d.cpu import Fields, GetFields, SetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpi:
    def __init__(self, fields, target_rank, tmax):
        common.check_type('fields', fields, Fields)

        # local variables
        nx, ny, nz = fields.ns
        dtype = fields.dtype
        mpi_type = fields.mpi_type

        common.check_value('mpi_type', mpi_type, \
                ['x+', 'x-', 'y+', 'y-', 'z+', 'z-'])

        # create instances (getf, setf and mpi requests)
        if '+' in mpi_type:     # split h
            getf = GetFields(fields, ['hy', 'hz'], \
                    (1, 1, 1), (1, ny-1, nz-1))
            setf = SetFields(fields, ['ey', 'ez'], \
                    (nx-1, 0, 0), (nx-1, ny-2, nz-2), True)

            req_send = comm.Send_init(getf.host_array, target_rank, tag=1)
            tmp_recv = np.zeros(getf.host_array.shape, dtype)
            req_recv = comm.Recv_init(tmp_recv, target_rank, tag=2)

        elif '-' in mpi_type:   # split e
            getf = GetFields(fields, ['ey', 'ez'], \
                    (nx-2, 0, 0), (nx-2, ny-2, nz-2))
            setf = SetFields(fields, ['hy', 'hz'], \
                    (0, 1, 1), (0, ny-1, nz-1), True)

            req_send = comm.Send_init(getf.host_array, target_rank, tag=2)
            tmp_recv = np.zeros(getf.host_array.shape, dtype)
            req_recv = comm.Recv_init(tmp_recv, target_rank, tag=1)

        # global variables and functions
        self.mainf = fields
        self.getf = getf
        self.setf = setf
        self.tmp_recv = tmp_recv
        self.req_send = req_send
        self.req_recv = req_recv

        self.tmax = tmax
        self.tstep = 1

        # append to the update list
        self.priority_type = 'mpi'
        self.mainf.append_instance(self)


    def update_e(self, part=''):
        if part == '':
            if self.tstep == 1:
                self.mainf.enqueue(self.req_recv.Start)
            else:
                self.mainf.enqueue(self.req_send.Wait)

        elif part == 'pre':
            evt = self.getf.get_event()
            self.mainf.enqueue(self.req_send.Start, wait_for=[evt])

        elif part == 'mid' and self.tstep != 1:
            self.mainf.enqueue(self.req_recv.Wait)
            self.setf.set_fields(self.tmp_recv)

        elif part == 'post':
            self.mainf.enqueue(self.req_recv.Start)


    def update_h(self, part=''):
        if part == '':
            self.mainf.enqueue(self.req_send.Wait)
            self.tstep += 1

        elif part == 'pre' and self.tstep != self.tmax:
            evt = self.getf.get_event()
            self.mainf.enqueue(self.req_send.Start, wait_for=[evt])

        elif part == 'mid':
            self.mainf.enqueue(self.req_recv.Wait)
            self.setf.set_fields(self.tmp_recv)

        elif part == 'post' and self.tstep != self.tmax:
            self.mainf.enqueue(self.req_recv.Start)
            self.tstep += 1
