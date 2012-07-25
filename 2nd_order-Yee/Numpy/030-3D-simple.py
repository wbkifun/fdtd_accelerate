#!/usr/bin/env python

import numpy as np
from scipy.constants import physical_constants
C = physical_constants['speed of light in vacuum'][0]


def update_h(ex, ey, ez, hx, hy, hz):
	hx[:,1:,1:] -= 0.5*(ez[:,1:,1:] - ez[:,:-1,1:] - ey[:,1:,1:] + ey[:,1:,:-1])
	hy[1:,:,1:] -= 0.5*(ex[1:,:,1:] - ex[1:,:,:-1] - ez[1:,:,1:] + ez[:-1,:,1:])
	hz[1:,1:,:] -= 0.5*(ey[1:,1:,:] - ey[:-1,1:,:] - ex[1:,1:,:] + ex[1:,:-1,:])


def update_e(ex, ey, ez, hx, hy, hz, cex, cey, cez):
	ex[:,:-1,:-1] += cex[:,:-1,:-1]*(hz[:,1:,:-1] - hz[:,:-1,:-1] - hy[:,:-1,1:] + hy[:,:-1,:-1])
	ey[:-1,:,:-1] += cey[:-1,:,:-1]*(hx[:-1,:,1:] - hx[:-1,:,:-1] - hz[1:,:,:-1] + hz[:-1,:,:-1])
	ez[:-1,:-1,:] += cez[:-1,:-1,:]*(hy[1:,:-1,:] - hy[:-1,:-1,:] - hx[:-1,1:,:] + hx[:-1,:-1,:])


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

# main loop
for tstep in xrange(1, tmax+1):
	update_h(*em_arrays)
	hx[:,:,0] = hx[:,:,-1]
	hy[:,:,0] = hy[:,:,-1]

	update_e(*(em_arrays + ce_arrays))
	ez[nx/2,ny/2,1] += np.sin(2*np.pi*frequency*dt*tstep)
	ex[:,:,-1] = ex[:,:,0]
	ey[:,:,-1] = ey[:,:,0]

	if tstep%tgap == 0:
		print 'tstep=%d' % tstep
		im.set_array(ez[:,:,nz/2].T**2)
		plt.draw()
		#savefig('./png/%.5d.png' % tstep) 
