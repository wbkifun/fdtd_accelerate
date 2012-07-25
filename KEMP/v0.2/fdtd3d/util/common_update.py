import numpy as np


def update_e(ehs, ces):
    ex, ey, ez, hx, hy, hz = ehs
    cex, cey, cez = ces

    ex[:, :-1, :-1] += cex[:, :-1, :-1] * \
            ((hz[:, 1:, :-1] - hz[:, :-1, :-1]) - (hy[:, :-1, 1:] - hy[:, :-1, :-1]))
    ey[:-1, :, :-1] += cey[:-1, :, :-1] * \
            ((hx[:-1, :, 1:] - hx[:-1, :, :-1]) - (hz[1:, :, :-1] - hz[:-1, :, :-1]))
    ez[:-1, :-1, :] += cez[:-1, :-1, :] * \
            ((hy[1:, :-1, :] - hy[:-1, :-1, :]) - (hx[:-1, 1:, :] - hx[:-1, :-1, :]))



def update_h(ehs, chs):
    ex, ey, ez, hx, hy, hz = ehs
    chx, chy, chz = chs

    hx[:, 1:, 1:] -= chx[:, 1:, 1:] * \
            ((ez[:, 1:, 1:] - ez[:, :-1, 1:]) - (ey[:, 1:, 1:] - ey[:, 1:, :-1]))
    hy[1:, :, 1:] -= chy[1:, :, 1:] * \
            ((ex[1:, :, 1:] - ex[1:, :, :-1]) - (ez[1:, :, 1:] - ez[:-1, :, 1:]))
    hz[1:, 1:, :] -= chz[1:, 1:, :] * \
            ((ey[1:, 1:, :] - ey[:-1, 1:, :]) - (ex[1:, 1:, :] - ex[1:, :-1, :]))



def generate_random_ehs(nx, ny, nz, dtype, ufunc=''):
    ns = (nx, ny, nz)

    if ufunc == 'e':
        ehs = [np.zeros(ns, dtype=dtype) for i in range(3)] + \
                [np.random.rand(*ns).astype(dtype) for i in range(3)]

    elif ufunc == 'h':
        ehs = [np.random.rand(*ns).astype(dtype) for i in range(3)] + \
                [np.zeros(ns, dtype=dtype) for i in range(3)]

    elif ufunc == '':
        ehs = [np.random.rand(*ns).astype(dtype) for i in range(6)]

    return ehs



def generate_random_cs(coeff_use, nx, ny, nz, dtype):
    ns = (nx, ny, nz)

    if 'e' in coeff_use:
        ces = [np.random.rand(*ns).astype(dtype) for i in range(3)]
    else:
        ces = [np.ones(ns, dtype=dtype)*0.5 for i in range(3)]

    if 'h' in coeff_use:
        chs = [np.random.rand(*ns).astype(dtype) for i in range(3)]
    else:
        chs = [np.ones(ns, dtype=dtype)*0.5 for i in range(3)]

    return (ces, chs)
