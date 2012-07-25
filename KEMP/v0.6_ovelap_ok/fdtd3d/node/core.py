from kemp.fdtd3d.util import common
from kemp.fdtd3d import cpu

from fields import Fields


class Core:
    def __init__(self, node_fields):

        common.check_type('node_fields', node_fields, Fields)

        # create Core instances
        f_list = node_fields.updatef_list

        self.cpu = cpu
        if 'gpu' in [f.device_type for f in f_list]:
            from kemp.fdtd3d import gpu
            self.gpu = gpu

        for f in f_list:
            getattr(self, f.device_type).Core(f)
