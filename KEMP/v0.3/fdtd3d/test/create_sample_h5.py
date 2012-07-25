import h5py as h5

f = h5.File('geo.h5', 'w')
f.attrs['nx'] = 150
f.attrs['ny'] = 140
f.attrs['nz'] = 64
f.attrs['coeff_use'] = ''
f.close()
