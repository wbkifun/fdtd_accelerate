import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu, cpu, node


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiBlock():
    def __init__(self, nodef, direction):
        common.check_type('nodef', nodef, node.Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        self.gpu = gpu
        self.cpu = cpu

        if '+' in direction:
            mf_p = nodef.mainf_list[-1]
            mpu = getattr(self, mf_p.device_type)
            
            self.getf_h = mpu.GetFields(mf_p, ['hy', 'hz'], (-1, 0, 0), (-1, -1, -1)) 
            self.setf_e = mpu.SetFields(mf_p, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.tmp_recv_e = np.zeros(self.getf_h.host_array.shape, nodef.dtype)

        if '-' in direction:
            mf_m = nodef.mainf_list[0]
            mpu = getattr(self, mf_m.device_type)
            
            self.getf_e = mpu.GetFields(mf_m, ['ey', 'ez'], (0, 0, 0), (0, -1, -1)) 
            self.setf_h = mpu.SetFields(mf_m, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.tmp_recv_h = np.zeros(self.getf_e.host_array.shape, nodef.dtype)

        # global variables
        self.direction = direction
        self.target_p = 0 if rank == size-1 else rank+1
        self.target_m = size-1 if rank == 0 else rank-1

        # append to the update list
        self.priority_type = 'mpi'
        nodef.append_instance(self)



    def update_e(self):
        if rank == 0:
            comm.Recv(self.tmp_recv_e, self.target_p, 1)
            self.setf_e.set_fields(self.tmp_recv_e)
            self.getf_e.get_event().wait()
            comm.Send(self.getf_e.get_fields(), self.target_m, 1)
        else:
            self.getf_e.get_event().wait()
            comm.Send(self.getf_e.get_fields(), self.target_m, 1)
            comm.Recv(self.tmp_recv_e, self.target_p, 1)
            self.setf_e.set_fields(self.tmp_recv_e)


    def update_h(self):
        if rank == 0:
            comm.Recv(self.tmp_recv_h, self.target_m, 0)
            self.setf_h.set_fields(self.tmp_recv_h)
            self.getf_h.get_event().wait()
            comm.Send(self.getf_h.get_fields(), self.target_p, 0)
        else:
            self.getf_h.get_event().wait()
            comm.Send(self.getf_h.get_fields(), self.target_p, 0)
            comm.Recv(self.tmp_recv_h, self.target_m, 0)
            self.setf_h.set_fields(self.tmp_recv_h)
