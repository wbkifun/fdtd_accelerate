import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common, common_exchange
from kemp.fdtd3d import gpu, cpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class BufferFields(cpu.Fields):
    def __init__(self, gpuf, direction, tmax, ny, nz, coeff_use, precision_float):
        """
        """

        super(BufferFields, self).__init__(3, ny, nz, coeff_use, precision_float, use_cpu_core=0)
        common.check_type('gpuf', gpuf, gpu.Fields)
        common.check_value('direction', direction, ('x+', 'x-'))

        if direction == 'x+':
            gf0 = gpu.GetFields(gpuf, ['hy', 'hz'], (-2, 0, 0), (-2, -1, -1)) 
            sf0 = cpu.SetFields(self, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            gf1 = cpu.GetFields(self, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            sf1 = gpu.SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            gf2 = cpu.GetFields(self, ['hy', 'hz'], (1, 0, 0), (1, -1, -1)) 
            sf2 = cpu.SetFields(self, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True) 

            target_rank = rank + 1
            tag_send, tag_recv = 0, 1

        elif direction == 'x-':
            gf0 = gpu.GetFields(gpuf, ['ey', 'ez'], (2, 0, 0), (2, -1, -1)) 
            sf0 = cpu.SetFields(self, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            gf1 = cpu.GetFields(self, ['hy', 'hz'], (1, 0, 0), (1, -1, -1)) 
            sf1 = gpu.SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            gf2 = cpu.GetFields(self, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            sf2 = cpu.SetFields(self, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True) 

            target_rank = rank - 1
            tag_send, tag_recv = 1, 0

        req_send = comm.Send_init(gf2.host_array, target_rank, tag=tag_send)
        tmp_recv_list = [np.zeros(gf2.host_array.shape, gpuf.dtype) for i in range(2)]
        req_recv_list = [comm.Recv_init(tmp_recv, target_rank, tag=tag_recv) for tmp_recv in tmp_recv_list]

        # global variables
        self.direction = direction
        self.gf0 = gf0
        self.sf0 = sf0
        self.gf1 = gf1
        self.sf1 = sf1
        self.gf2 = gf2
        self.sf2 = sf2
        self.req_send = req_send
        self.req_recv_list = req_recv_list
        self.tmp_recv_list = tmp_recv_list

        self.switch = 0
        self.tmax = tmax
        self.tstep = 1

        # global functions
        if direction == 'x+':
            self.update_e = self.update_e_xp
            self.update_h = self.update_h_xp
        elif direction == 'x-':
            self.update_e = self.update_e_xm
            self.update_h = self.update_h_xm



    def update_e_xp(self):
        # mpi recv
        if self.tstep == 1:
            self.req_recv_list[self.switch].Start()

        # internal recv h
        self.sf0.set_fields(self.gf0.get_fields(), [self.gf0.get_event()])

        # update and internal send e
        for instance in self.instance_list:
            instance.update_e()
        self.gf1.get_event().wait()
        self.sf1.set_fields(self.gf1.get_fields())
        #self.enqueue(self.sf1.set_fields, [self.gf1.get_fields()], [self.gf1.get_event()])



    def update_h_xp(self):
        # update pre
        for instance in self.instance_list:
            instance.update_h('pre')

        # mpi send h
        if self.tstep > 1:
            self.req_send.Wait()
            
        if self.tstep < self.tmax:
            self.gf2.get_event().wait()
            self.req_send.Start()

        # mpi recv e
        self.req_recv_list[self.switch].Wait()
        self.sf2.set_fields(self.tmp_recv_list[self.switch])
        self.switch = 1 if self.switch == 0 else 0

        if self.tstep < self.tmax:
            self.req_recv_list[self.switch].Start()

        # update post
        for instance in self.instance_list:
            instance.update_h('post')

        self.tstep += 1


    def update_e_xm(self):
        # update pre
        for instance in self.instance_list:
            instance.update_e('pre')

        # mpi send e
        if self.tstep > 1:
            self.req_send.Wait()

        self.gf2.get_event().wait()
        self.req_send.Start()

        # mpi recv h
        if self.tstep > 1:
            self.req_recv_list[self.switch].Wait()
            self.sf2.set_fields(self.tmp_recv_list[self.switch])
            self.switch = 1 if self.switch == 0 else 0

        if self.tstep < self.tmax:
            self.req_recv_list[self.switch].Start()

        # update post
        for instance in self.instance_list:
            instance.update_e('post')



    def update_h_xm(self):
        # mpi recv
        if self.tstep == self.tmax:
            self.req_send.Wait()

        # internal recv e
        self.sf0.set_fields(self.gf0.get_fields(), [self.gf0.get_event()])

        # update and internal send h
        for instance in self.instance_list:
            instance.update_h()
        self.gf1.get_event().wait()
        self.sf1.set_fields(self.gf1.get_fields())
        #self.enqueue(self.sf1.set_fields, [self.gf1.get_fields()], [self.gf1.get_event()])

        self.tstep += 1
