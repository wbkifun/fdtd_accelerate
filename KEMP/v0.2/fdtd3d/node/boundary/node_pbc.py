import numpy as np

from kemp.fdtd3d.util import common
from kemp.fdtd3d.node import NodeFields
from kemp.fdtd3d import gpu, cpu


class NodePbc:
    def __init__(self, node_fields, axis):
        """
        """

        common.check_type('node_fields', node_fields, NodeFields)
        common.check_value('axis', axis, ['x', 'y', 'z'])

        # local variables
        self.gpu = gpu
        self.cpu = cpu
        mainf_list = node_fields.mainf_list
        cpuf_dict = node_fields.cpuf_dict
        axis_id = {'x':0, 'y':1, 'z':2}[axis]

        set_cpuf = set( cpuf_dict.keys() )
        for ax in ['x', 'y', 'z']:
            if not set_cpuf.isdisjoint( [ax+'+', ax+'-'] ):
                raise ValueError, 'There are %s-axis buffer instances. The pbc_internal operation along %s-axis is not allowed.' % (ax, ax)

        # create pbc instances
        f0 = mainf_list[0]
        f1 = mainf_list[-1]

        if axis == 'x':
            if f0 is not f1:
                setf_e = cpu.SetFields(f1, ['ey', 'ez'], \
                        (f1.nx-1, 0, 0), (f1.nx-1, f1.ny-2, f1.nz-2), True)
                getf_e = gpu.GetFields(f0, ['ey', 'ez'], \
                        (0, 0, 0), (0, f0.ny-2, f0.nz-2) )

                setf_h = gpu.SetFields(f0, ['hy', 'hz'], \
                        (0, 1, 1), (0, f0.ny-1, f0.nz-1), True )
                getf_h = cpu.GetFields(f1, ['hy', 'hz'], \
                        (f1.nx-1, 1, 1), (f1.nx-1, f1.ny-1, f1.nz-1) )

            else:
                getattr(self, f0.device_type).Pbc(f0, axis)

        elif axis in ['y', 'z']:
            for f in mainf_list:
                getattr(self, f.device_type).Pbc(f, axis)

        # global variables and functions
        if axis == 'x' and f0 is not f1:
            self.setf_e = setf_e
            self.getf_e = getf_e
            self.setf_h = setf_h
            self.getf_h = getf_h

            self.update_e = self.update_e_actual
            self.update_h = self.update_h_actual

        else:
            self.update_e = lambda : None
            self.update_h = lambda : None


    def update_e_actual(self):
        self.setf_e.set_fields(self.getf_e.get_fields(), \
                [self.getf_e.get_event()])


    def update_h_actual(self):
        self.getf_h.get_event().wait()
        self.setf_h.set_fields( self.getf_h.get_fields() )
