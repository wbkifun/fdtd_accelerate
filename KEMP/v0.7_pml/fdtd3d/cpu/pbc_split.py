from kemp.fdtd3d.util import common, common_exchange, common_buffer
from fields import Fields


class Pbc:
    def __init__(self, fields, axes):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_type('axes', axes, str)

        assert len( set(axes).intersection(set('xyz')) ) > 0, 'axes option is wrong: %s is given' % repr(axes)

        # global variables
        self.mainf = fields
        self.axes = axes

        # append to the update list
        self.priority_type = 'pbc'
        self.mainf.append_instance(self)


    def update(self, e_or_h, part):
        for axis in list(self.axes):
            for str_f in common_exchange.str_fs_dict[axis][e_or_h]:
                f = self.mainf.get(str_f)
                sl_get = common_exchange.slice_dict[axis][e_or_h]['get']
                sl_set = common_exchange.slice_dict[axis][e_or_h]['set']
                sl_part = common_buffer.slice_dict[e_or_h][part]

                sl_src = common.overlap_two_slices(self.mainf.ns, sl_get, sl_part)
                sl_dest = common.overlap_two_slices(self.mainf.ns, sl_set, sl_part)
                f[sl_dest] = f[sl_src]


    def update_e(self, part=''):
        self.mainf.enqueue(self.update, ['e', part])


    def update_h(self, part=''):
        self.mainf.enqueue(self.update, ['h', part])
