import numpy as np

from kemp.fdtd3d.util import common
from kemp.fdtd3d import cpu


class Fields:
    def __init__(self, mainf_list, buffer_dict={}):
        """
        """

        try:
            from kemp.fdtd3d import gpu
            common.check_type('mainf_list', mainf_list, (list, tuple), (gpu.Fields, cpu.Fields))
        except:
            common.check_type('mainf_list', mainf_list, (list, tuple), cpu.Fields)
        common.check_type('buffer_dict', buffer_dict, dict)

        # local variables
        device_type_list = [f.device_type for f in mainf_list]
        if 'cpu' in device_type_list:
            cpuf = mainf_list[ device_type_list.index('cpu') ]
        else:
            cpuf = None

        nx_list = [f.nx for f in mainf_list]
        nx = int( sum(nx_list) - len(nx_list) + 1 )
        ny, nz = [int(n) for n in mainf_list[0].ns[1:]]
        accum_nx_list = np.add.accumulate([0] + [f.nx-1 for f in mainf_list])
        #accum_nx_list[-1] += 1
        accum_nx_list = [int(anx) for anx in accum_nx_list]

        # global variables
        self.mainf_list = mainf_list
        self.buffer_dict = buffer_dict
        self.updatef_list = mainf_list[:] + buffer_dict.values()
        self.cpuf = cpuf

        self.dtype = mainf_list[0].dtype

        self.nx = nx
        self.nx_list = nx_list
        self.accum_nx_list = accum_nx_list
        self.ns = (nx, ny, nz)

        # update list
        self.instance_list = []
        self.append_instance = lambda instance: \
                common.append_instance(self.instance_list, instance)


    def update_e(self):
        for f in self.updatef_list:
            f.update_e()

        for instance in self.instance_list:
            instance.update_e()


    def update_h(self):
        for f in self.updatef_list:
            f.update_h()

        for instance in self.instance_list:
            instance.update_h()
