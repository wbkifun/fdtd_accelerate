from kemp.fdtd3d.util import common, common_exchange
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
        self.ns = fields.ns

        # append to the update list
        self.priority_type = 'pbc'
        self.mainf.append_instance(self)


    def update(self, e_or_h):
        for axis in list(self.axes):
            for str_f in common_exchange.str_fs_dict[axis][e_or_h]:
                sl_get = common_exchange.slice_dict(*self.ns)[axis][e_or_h]['get']
                sl_set = common_exchange.slice_dict(*self.ns)[axis][e_or_h]['set']
                f = self.mainf.get(str_f)
                f[sl_set] = f[sl_get]


    def update_e(self):
        self.mainf.enqueue(self.update, ['e'])


    def update_h(self):
        self.mainf.enqueue(self.update, ['h'])
