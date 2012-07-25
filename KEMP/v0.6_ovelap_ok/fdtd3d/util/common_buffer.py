pt0_dict = { \
        'e': {'pre': (1, 0, 0), 'post': (0, 0, 0)}, \
        'h': {'pre': (0, 0, 0), 'post': (2, 0, 0)} }

pt1_dict = lambda ny, nz: { \
        'e': {'pre': (2, ny-1, nz-1), 'post': (0, ny-1, nz-1)}, \
        'h': {'pre': (1, ny-1, nz-1), 'post': (2, ny-1, nz-1)} }


sl0 = slice(None, None)
slice_dict = { \
        'e': {'': (sl0, sl0, sl0), 'pre': (slice(1, None), sl0, sl0), 'post': (slice(0, 1), sl0, sl0) }, \
        'h': {'': (sl0, sl0, sl0), 'pre': (slice(None, 2), sl0, sl0), 'post': (slice(2, 3), sl0, sl0) } }


# axes_dict[buffer.direction[0]][axis]
axes_dict = { \
        'x': {'x': 'x', 'y': 'y', 'z': 'z'}, \
        'y': {'x': 'y', 'y': 'x', 'z': 'z'}, \
        'z': {'x': 'y', 'y': 'z', 'z': 'x'} }
