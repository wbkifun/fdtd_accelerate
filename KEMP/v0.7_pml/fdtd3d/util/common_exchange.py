str_fs_dict = { \
        'x': {'e': ['ey','ez'], 'h': ['hy','hz']}, \
        'y': {'e': ['ex','ez'], 'h': ['hx','hz']}, \
        'z': {'e': ['ex','ey'], 'h': ['hx','hy']} }

pt0_dict = lambda nx, ny, nz: { \
        'x': {'e': {'get': (0, 0, 0),    'set': (nx-1, 0, 0)}, \
              'h': {'get': (nx-1, 0, 0), 'set': (0, 0, 0)   } }, \
        'y': {'e': {'get': (0, 0, 0),    'set': (0, ny-1, 0)}, \
              'h': {'get': (0, ny-1, 0), 'set': (0, 0, 0)   } }, \
        'z': {'e': {'get': (0, 0, 0),    'set': (0, 0, nz-1)}, \
              'h': {'get': (0, 0, nz-1), 'set': (0, 0, 0)   } } }

pt1_dict = lambda nx, ny, nz: { \
        'x': {'e': {'get': (0, ny-1, nz-1),    'set': (nx-1, ny-1, nz-1)}, \
              'h': {'get': (nx-1, ny-1, nz-1), 'set': (0, ny-1, nz-1)   } }, \
        'y': {'e': {'get': (nx-1, 0, nz-1),    'set': (nx-1, ny-1, nz-1)}, \
              'h': {'get': (nx-1, ny-1, nz-1), 'set': (nx-1, 0, nz-1)   } }, \
        'z': {'e': {'get': (nx-1, ny-1, 0),    'set': (nx-1, ny-1, nz-1)}, \
              'h': {'get': (nx-1, ny-1, nz-1), 'set': (nx-1, ny-1, 0)   } } }

sl = slice(None, None)
slice_dict = lambda nx, ny, nz: { \
        'x': {'e': {'get': (0, sl, sl), 'set': (nx-1, sl, sl)}, \
              'h': {'get': (nx-1, sl, sl), 'set': (0, sl, sl)} }, \
        'y': {'e': {'get': (sl, 0, sl), 'set': (sl, ny-1, sl)}, \
              'h': {'get': (sl, ny-1, sl), 'set': (sl, 0, sl)} }, \
        'z': {'e': {'get': (sl, sl, 0), 'set': (sl, sl, nz-1)}, \
              'h': {'get': (sl, sl, nz-1), 'set': (sl, sl, 0)} } }
