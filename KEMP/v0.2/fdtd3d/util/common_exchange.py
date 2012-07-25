import numpy as np

from kemp.fdtd3d import gpu, cpu



str_fs_dict = { \
        'x': {'e': ['ey','ez'], 'h': ['hy','hz']}, \
        'y': {'e': ['ex','ez'], 'h': ['hx','hz']}, \
        'z': {'e': ['ex','ey'], 'h': ['hx','hy']} }

pt0_dict = lambda nx, ny, nz: { \
        'x': {'e': {'get': (0, 0, 0), 'set': (nx-1, 0, 0)}, \
              'h': {'get': (nx-1, 1, 1), 'set': (0, 1, 1)} }, \
        'y': {'e': {'get': (0, 0, 0), 'set': (0, ny-1, 0)}, \
              'h': {'get': (1, ny-1, 1), 'set': (1, 0, 1)} }, \
        'z': {'e': {'get': (0, 0, 0), 'set': (0, 0, nz-1)}, \
              'h': {'get': (1, 1, nz-1), 'set': (1, 1, 0)} } }

pt1_dict = lambda nx, ny, nz: { \
        'x': {'e': {'get': (0, ny-2, nz-2), 'set': (nx-1, ny-2, nz-2)}, \
              'h': {'get': (nx-1, ny-1, nz-1), 'set': (0, ny-1, nz-1)} }, \
        'y': {'e': {'get': (nx-2, 0, nz-2), 'set': (nx-2, ny-1, nz-2)}, \
              'h': {'get': (nx-1, ny-1, nz-1), 'set': (nx-1, 0, nz-1)} }, \
        'z': {'e': {'get': (nx-2, ny-2, 0), 'set': (nx-2, ny-2, nz-1)}, \
              'h': {'get': (nx-1, ny-1, nz-1), 'set': (nx-1, ny-1, 0)} } }



class VerifyPbc:
    def __init__(self, fields, axes):
        # local variables
        nx, ny, nz = fields.ns
        self.gpu, self.cpu = gpu, cpu
        getf0_dict = {}
        getf1_dict = {}

        # create the GetFields instances
        for axis in axes:
            getf0_dict[axis] = {}
            getf1_dict[axis] = {}

            for e_or_h in ['e', 'h']:
                str_fs = str_fs_dict[axis][e_or_h]
                pt0 = pt0_dict(nx, ny, nz)[axis][e_or_h]
                pt1 = pt1_dict(nx, ny, nz)[axis][e_or_h]
                getf0_dict[axis][e_or_h] = \
                        getattr(self, fields.device_type).GetFields( \
                        fields, str_fs, pt0['get'], pt1['get'])
                getf1_dict[axis][e_or_h] = \
                        getattr(self, fields.device_type).GetFields( \
                        fields, str_fs, pt0['set'], pt1['set'])

        # global variabels
        self.axes = axes
        self.getf0_dict = getf0_dict
        self.getf1_dict = getf1_dict


    def verify_e(self):
        for axis in self.axes:
            g0 = self.getf0_dict[axis]['e']
            g1 = self.getf1_dict[axis]['e']
            g0.get_event().wait()
            g1.get_event().wait()
            norm = np.linalg.norm(g0.get_fields() - g1.get_fields())
            assert norm == 0, 'e, axis= %s, norm= %g' % (axis, norm)


    def verify_h(self):
        for axis in self.axes:
            g0 = self.getf0_dict[axis]['h']
            g1 = self.getf1_dict[axis]['h']
            g0.get_event().wait()
            g1.get_event().wait()
            norm = np.linalg.norm(g0.get_fields() - g1.get_fields())
            assert norm == 0, 'e, axis= %s, norm= %g' % (axis, norm)
