import numpy as np

from kemp.fdtd3d.util import common, common_cpu
from fields import Fields


class BufferFields(Fields):
    def __init__(self, direction, target_rank, ny, nz, coeff_use, precision_float):
        """
        """

        super(BufferFields, self).__init__(3, ny, nz, coeff_use, precision_float, use_cpu_core=0)
        common.check_value('direction', direction, ('x+', 'x-', 'y+', 'y-', 'z+', 'z-'))
        common.check_type('target_rank', target_rank, int)

        # global variables
        self.direction = direction
        self.target_rank = target_rank

        p_or_m = direction[-1]
        self.part_e_list = {'+': [''], '-': ['pre', 'post']}[p_or_m]
        self.part_h_list = {'-': [''], '+': ['pre', 'post']}[p_or_m]

        self.is_split_dict = { \
                '+': {'e': False, 'h': True}, \
                '-': {'e': True, 'h': False}}[p_or_m]


    def update(self, e_or_h, part):
        if self.is_split_dict[e_or_h]:
            for instance in self.instance_list:
                getattr(instance, 'update_%s' % e_or_h)(part)

        elif part == 'pre':
            for instance in self.instance_list:
                getattr(instance, 'update_%s' % e_or_h)('')


    def update_e(self, part):
        self.update('e', part)


    def update_h(self, part):
        self.update('h', part)
