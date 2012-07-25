#!/usr/bin/env python

import subprocess as sp


cmd_memcpy = './343-measure-memcpy-sdk-fit.py bandwidth_PCIE.g101.h5'
cmd_mpi = './353-measure-mpi-fit.py bandwidth_GbE.h5'

bl = {}		# bandwidth, latency

out, err = sp.Popen(cmd_mpi.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
vals = out.split()
bl['nton'] = float(vals[1]), float(vals[2])

out, err = sp.Popen(cmd_memcpy.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
lines = out.splitlines()
for line in lines:
	vals = line.split()
	bl[vals[0]] = float(vals[1]), float(vals[2])

for key, val in bl.items():
	print('%s\t%g\t%g' % (key, val[0], val[1]))


# Estimate the elapsed time for FDTD
import numpy as np
from utils import get_nbytes_unit

bpc = {'update_e':64.25, 'update_h':52.25}	# bytes / cell
et = lambda (bw, lt), nbytes: nbytes / bw + lt	# elapsed time

nx, ny, nz = (480, 480, 480)
print('')
print('(%d, %d, %d)' % (nx, ny, nz)),
print(' %1.2f %s' % get_nbytes_unit(nx*ny*nz*9*4))
print('%s\t%g' % ('update_e', et(bl['dtod'], bpc['update_e'] * nx*ny*nz)))
print('%s\t%g' % ('update_h', et(bl['dtod'], bpc['update_h'] * nx*ny*nz)))
print('%s\t%g' % ('dtoh,paged', et(bl['dtoh,pageable'], np.nbytes['float32'] * 2*ny*nz)))
print('%s\t%g' % ('htod,paged', et(bl['htod,pageable'], np.nbytes['float32'] * 2*ny*nz)))
print('%s\t\t%g' % ('nton', et(bl['nton'], np.nbytes['float32'] * 2*ny*nz)))

nx, ny, nz = (256, 256, 256)
print('')
print('(%d, %d, %d)' % (nx, ny, nz)),
print(' %1.2f %s' % get_nbytes_unit(nx*ny*nz*9*4))
print('%s\t%g' % ('update_e', et(bl['dtod'], bpc['update_e'] * nx*ny*nz)))
print('%s\t%g' % ('update_h', et(bl['dtod'], bpc['update_h'] * nx*ny*nz)))
print('%s\t%g' % ('dtoh,paged', et(bl['dtoh,pageable'], np.nbytes['float32'] * 2*ny*nz)))
print('%s\t%g' % ('htod,paged', et(bl['htod,pageable'], np.nbytes['float32'] * 2*ny*nz)))
print('%s\t\t%g' % ('nton', et(bl['nton'], np.nbytes['float32'] * 2*ny*nz)))

nx, ny, nz = (96, 96, 96)
print('')
print('(%d, %d, %d)' % (nx, ny, nz)),
print(' %1.2f %s' % get_nbytes_unit(nx*ny*nz*9*4))
print('%s\t%g' % ('update_e', et(bl['dtod'], bpc['update_e'] * nx*ny*nz)))
print('%s\t%g' % ('update_h', et(bl['dtod'], bpc['update_h'] * nx*ny*nz)))
print('%s\t%g' % ('dtoh,paged', et(bl['dtoh,pageable'], np.nbytes['float32'] * 2*ny*nz)))
print('%s\t%g' % ('htod,paged', et(bl['htod,pageable'], np.nbytes['float32'] * 2*ny*nz)))
print('%s\t\t%g' % ('nton', et(bl['nton'], np.nbytes['float32'] * 2*ny*nz)))
