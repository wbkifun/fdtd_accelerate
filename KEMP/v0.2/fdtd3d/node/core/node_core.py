from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu, cpu

from node_fields import NodeFields


class NodeCore:
    def __init__(self, node_fields):

        common.check_type('node_fields', node_fields, NodeFields)

        # local variables
        nodef = node_fields
        updatef_list = nodef.updatef_list

        # create Core instances
        self.gpu, self.cpu = gpu, cpu
        core_list = []
        for f in updatef_list:
            core_list.append( getattr(self, f.device_type).Core(f) )

        # global variables
        self.core_list = core_list
