import numpy as np

from kemp.fdtd3d.util import common, common_buffer
from kemp.fdtd3d.node import Fields
from kemp.fdtd3d import cpu


class Pbc:
    def __init__(self, node_fields, axes):
        """
        """

        common.check_type('node_fields', node_fields, Fields)
        common.check_type('axes', axes, str)

        assert len( set(axes).intersection(set('xyz')) ) > 0, 'axes option is wrong: %s is given' % repr(axes)

        # local variables
        nodef = node_fields
        mainf_list = nodef.mainf_list
        buffer_dict = nodef.buffer_dict

        self.cpu = cpu
        if 'gpu' in [f.device_type for f in nodef.updatef_list]:
            from kemp.fdtd3d import gpu
            self.gpu = gpu

        # create pbc instances
        f0 = mainf_list[-1]
        f1 = mainf_list[0]

        if f0 is f1:
            getattr(self, f0.device_type).Pbc(f0, axes)

        else:
            if 'x' in axes:
                nx, ny, nz = f0.ns

                self.getf_e = getattr(self, f1.device_type).GetFields(f1, ['ey', 'ez'], (0, 0, 0), (0, ny-1, nz-1) )
                self.setf_e = getattr(self, f0.device_type).SetFields(f0, ['ey', 'ez'], (nx-1, 0, 0), (nx-1, ny-1, nz-1), True)

                self.getf_h = getattr(self, f0.device_type).GetFields(f0, ['hy', 'hz'], (nx-1, 0, 0), (nx-1, ny-1, nz-1) )
                self.setf_h = getattr(self, f1.device_type).SetFields(f1, ['hy', 'hz'], (0, 0, 0), (0, ny-1, nz-1), True )

                # append to the update list
                self.priority_type = 'pbc'
                nodef.append_instance(self)

            axs = axes.strip('x')
            if len( set(axs).intersection(set('yz')) ) > 0:
                for f in mainf_list:
                    getattr(self, f.device_type).Pbc(f, axs)

        # for buffer fields
        for direction , buffer in buffer_dict.items():
            replaced_axes = map(lambda ax: common_buffer.axes_dict[direction[0]][ax], list(axes))
            axs = ''.join(replaced_axes).strip('x')
            if axs != '':
                cpu.Pbc(buffer, axs)


    def update_e(self):
        self.getf_e.get_event().wait()
        self.setf_e.set_fields( self.getf_e.get_fields() )


    def update_h(self):
        self.setf_h.set_fields(self.getf_h.get_fields(), [self.getf_h.get_event()])
