import numpy as np

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu, cpu


class NodeFields:
    def __init__(self, mainf_list):
        """
        """

        common.check_type('mainf_list', mainf_list, \
                (list, tuple), (gpu.Fields, cpu.Fields))

        # local variables
        cpuf = mainf_list[-1]
        cpuf_dict = {cpuf.mpi_type: cpuf}

        nx_list = [f.nx for f in mainf_list]
        nx = sum(nx_list) - len(nx_list) + 1
        ny, nz = cpuf.ny, cpuf.nz

        accum_nx_list = [0] + \
                list( np.add.accumulate([f.nx-1 for f in mainf_list]) )

        # global variables
        self.mainf_list = mainf_list
        self.cpuf_dict = cpuf_dict
        self.updatef_list = mainf_list[:]

        self.dtype = cpuf.dtype
        self.nx = nx
        self.nx_list = nx_list
        self.accum_nx_list = accum_nx_list
        self.ns = (nx, ny, nz)


    def append_buffer_fields(self, cpuf):
        common.check_type('cpuf', cpuf, cpu.Fields)
        common.check_value('cpuf.mpi_type', cpuf.mpi_type, \
                ('x+', 'x-', 'y+', 'y-', 'z+', 'z-') )

        self.cpuf_dict[cpuf.mpi_type] = cpuf
        self.updatef_list.append(cpuf)


    def update_e(self):
        for f in self.updatef_list:
            f.update_e()


    def update_h(self):
        for f in self.updatef_list:
            f.update_h()
