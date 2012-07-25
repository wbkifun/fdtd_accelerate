#!/usr/bin/env python

import numpy as np
import sys
sys.path.append("./") 
from fdtd3d import update_h, update_e
from fdtd3d import update_pbc_h, update_pbc_e

from scipy.constants import physical_constants
C = physical_constants['speed of light in vacuum'][0]


# setup
nx, ny, nz = n = 200, 200, 2
tmax, tgap = 300, 10

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5
wavelength = 30*dx
frequency = C/wavelength

# allocate arrays
ex, ey, ez, hx, hy, hz = em_arrays = [np.zeros(n, 'f') for i in range(6)]
cex, cey, cez = ce_arrays= [np.ones(n, 'f')*0.5 for i in range(3)]

# prepare for plot
import matplotlib.pyplot as plt
plt.ion()
im = plt.imshow(np.ones((nx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()

# print memory usage
from util import print_mem
arrays_points = {'main':nx*ny*nz*9}
print_mem(n, np.nbytes['float32'], arrays_points)

# measure execution time
from util import time_dict, print_time
time_dict['point'] = nx*ny*nz
time_dict['flop'] = nx*ny*nz*30*tgap
time_dict['tmax'] = tmax
time_dict['tgap'] = tgap

# main loop
for tstep in xrange(1, tmax+1):
	update_h(*em_arrays)
	update_pbc_h('z', *em_arrays[3:])

	update_e(*(em_arrays + ce_arrays))
	ez[nx/2,ny/2,1] += np.sin(2*np.pi*frequency*dt*tstep)
	update_pbc_e('z', *em_arrays[:-3])

	if tstep%tgap == 0:
		print_time(tstep)
		im.set_array(ez[:,:,nz/2].T**2)
		plt.draw()
		#savefig('./png/%.5d.png' % tstep) 

print ''
