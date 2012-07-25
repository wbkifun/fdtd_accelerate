import numpy as np

from kemp.fdtd3d.util import common, common_cpu
from fields import Fields



class BufferFields(Fields):
    def __init__(self, direction, ny, nz, coeff_use, precision_float):
        """
        """

        super(BufferFields, self).__init__(3, ny, nz, coeff_use, precision_float, use_cpu_core=1)
        common.check_value('direction', direction, ('x+', 'x-', 'y+', 'y-', 'z+', 'z-'))

        # global variables
        self.direction = direction

        p_or_m = direction[-1]
        self.part_e_list = {'+': [''], '-': ['pre', 'post']}[p_or_m]
        self.part_h_list = {'-': [''], '+': ['pre', 'post']}[p_or_m]


    def update_e(self):
        for part in self.part_e_list:
            for instance in self.instance_list:
                instance.update_e(part)


    def update_h(self):
        for part in self.part_h_list:
            for instance in self.instance_list:
                instance.update_h(part)
