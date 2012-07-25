import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu, cpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiBufferBlock():
    def __init__(self, gpuf, direction):
        common.check_type('gpuf', gpuf, gpu.Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        qtask = cpu.QueueTask()

        if '+' in direction:
            self.cpuf_p = cpuf_p = cpu.Fields(qtask, 3, gpuf.ny, gpuf.nz, gpuf.coeff_use, gpuf.precision_float, use_cpu_core=0)

            self.gf_p_h = gpu.GetFields(gpuf, ['hy', 'hz'], (-2, 0, 0), (-2, -1, -1)) 
            self.sf_p_h = cpu.SetFields(cpuf_p, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.gf_p_e = cpu.GetFields(cpuf_p, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_p_e = gpu.SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.gf_h = cpu.GetFields(cpuf_p, ['hy', 'hz'], (1, 0, 0), (1, -1, -1)) 
            self.sf_e = cpu.SetFields(cpuf_p, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True) 

            self.tmp_recv_e = np.zeros(self.gf_h.host_array.shape, gpuf.dtype)

        if '-' in direction:
            self.cpuf_m = cpuf_m = cpu.Fields(qtask, 3, gpuf.ny, gpuf.nz, gpuf.coeff_use, gpuf.precision_float, use_cpu_core=0)
            self.gf_m_e = gpu.GetFields(gpuf, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_m_e = cpu.SetFields(cpuf_m, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.gf_m_h = cpu.GetFields(cpuf_m, ['hy', 'hz'], (1, 0, 0), (1, -1, -1)) 
            self.sf_m_h = gpu.SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.gf_e = cpu.GetFields(cpuf_m, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_h = cpu.SetFields(cpuf_m, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True) 

            self.tmp_recv_h = np.zeros(self.gf_e.host_array.shape, gpuf.dtype)

        # global variables
        self.direction = direction
        self.qtask = qtask



    def update_e(self):
        if '+' in self.direction:
            # update e
            for instance in self.cpuf_p.instance_list:
                instance.update_e()

            # internal send e
            self.gf_p_e.get_event().wait()
            self.sf_p_e.set_fields(self.gf_p_e.get_fields())
            #self.qtask.enqueue(self.sf_p_e.set_fields, [self.gf_p_e.get_fields()], [self.gf_p_e.get_event()])

            # mpi recv e
            comm.Recv(self.tmp_recv_e, rank+1, 1)
            self.sf_e.set_fields(self.tmp_recv_e)


        if '-' in self.direction:
            # update e
            for instance in self.cpuf_m.instance_list:
                instance.update_e()

            # internal recv e
            self.sf_m_e.set_fields(self.gf_m_e.get_fields(), [self.gf_m_e.get_event()])

            # mpi send e
            self.gf_e.get_event().wait()
            comm.Send(self.gf_e.get_fields(), rank-1, 1)



    def update_h(self):
        if '+' in self.direction:
            # update h
            for instance in self.cpuf_p.instance_list:
                instance.update_h()

            # internal recv h
            self.sf_p_h.set_fields(self.gf_p_h.get_fields(), [self.gf_p_h.get_event()])

            # mpi send h
            self.gf_h.get_event().wait()
            comm.Send(self.gf_h.get_fields(), rank+1, 0)


        if '-' in self.direction:
            # update h
            for instance in self.cpuf_m.instance_list:
                instance.update_h()

            # internal send h
            self.gf_m_h.get_event().wait()
            self.sf_m_h.set_fields(self.gf_m_h.get_fields())
            #self.qtask.enqueue(self.sf_m_h.set_fields, [self.gf_m_h.get_fields()], [self.gf_m_h.get_event()])

            # mpi recv h
            comm.Recv(self.tmp_recv_h, rank-1, 0)
            self.sf_h.set_fields(self.tmp_recv_h)
