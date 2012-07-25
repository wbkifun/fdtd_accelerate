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
sl0 = slice(0, 1)
sl1 = slice(-1, None)
slice_dict = { \
        'x': {'e': {'get': (sl0, sl, sl), 'set': (sl1, sl, sl)}, \
              'h': {'get': (sl1, sl, sl), 'set': (sl0, sl, sl)} }, \
        'y': {'e': {'get': (sl, sl0, sl), 'set': (sl, sl1, sl)}, \
              'h': {'get': (sl, sl1, sl), 'set': (sl, sl0, sl)} }, \
        'z': {'e': {'get': (sl, sl, sl0), 'set': (sl, sl, sl1)}, \
              'h': {'get': (sl, sl, sl1), 'set': (sl, sl, sl0)} } }

# for buffers
pt0_buf_dict = lambda nx, ny, nz: { \
        'x': {'e': {'get': (1, 0, 0),    'set': (nx-1, 0, 0)}, \
              'h': {'get': (nx-2, 0, 0), 'set': (0, 0, 0)   } }, \
        'y': {'e': {'get': (0, 1, 0),    'set': (0, ny-1, 0)}, \
              'h': {'get': (0, ny-2, 0), 'set': (0, 0, 0)   } }, \
        'z': {'e': {'get': (0, 0, 1),    'set': (0, 0, nz-1)}, \
              'h': {'get': (0, 0, nz-2), 'set': (0, 0, 0)   } } }

pt1_buf_dict = lambda nx, ny, nz: { \
        'x': {'e': {'get': (1, ny-1, nz-1),    'set': (nx-1, ny-1, nz-1)}, \
              'h': {'get': (nx-2, ny-1, nz-1), 'set': (0, ny-1, nz-1)   } }, \
        'y': {'e': {'get': (nx-1, 1, nz-1),    'set': (nx-1, ny-1, nz-1)}, \
              'h': {'get': (nx-1, ny-2, nz-1), 'set': (nx-1, 0, nz-1)   } }, \
        'z': {'e': {'get': (nx-1, ny-1, 1),    'set': (nx-1, ny-1, nz-1)}, \
              'h': {'get': (nx-1, ny-1, nz-2), 'set': (nx-1, ny-1, 0)   } } }

sl2 = slice(1, 2)
sl3 = slice(-2, -1)
slice_buf_dict = { \
        'x': {'e': {'get': (sl2, sl, sl), 'set': (sl1, sl, sl)}, \
              'h': {'get': (sl3, sl, sl), 'set': (sl0, sl, sl)} }, \
        'y': {'e': {'get': (sl, sl2, sl), 'set': (sl, sl1, sl)}, \
              'h': {'get': (sl, sl3, sl), 'set': (sl, sl0, sl)} }, \
        'z': {'e': {'get': (sl, sl, sl2), 'set': (sl, sl, sl1)}, \
              'h': {'get': (sl, sl, sl3), 'set': (sl, sl, sl0)} } }
