import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu, cpu, node


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiNonBlock():
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

            target = 0 if rank == size-1 else rank+1
            self.req_send_h = comm.Send_init(self.getf_h.host_array, target, tag=0)
            self.tmp_recv_e = np.zeros(self.getf_h.host_array.shape, nodef.dtype)
            self.req_recv_e = comm.Recv_init(self.tmp_recv_e, target, tag=1)

        if '-' in direction:
            mf_m = nodef.mainf_list[0]
            mpu = getattr(self, mf_m.device_type)
            
            self.getf_e = mpu.GetFields(mf_m, ['ey', 'ez'], (0, 0, 0), (0, -1, -1)) 
            self.setf_h = mpu.SetFields(mf_m, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            target = size-1 if rank == 0 else rank-1
            self.req_send_e = comm.Send_init(self.getf_e.host_array, target, tag=1)
            self.tmp_recv_h = np.zeros(self.getf_e.host_array.shape, nodef.dtype)
            self.req_recv_h = comm.Recv_init(self.tmp_recv_h, target, tag=0)

        # global variables
        self.direction = direction

        # append to the update list
        self.priority_type = 'mpi'
        nodef.append_instance(self)



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
