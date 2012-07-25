#!/usr/bin/env python

import numpy as np
import sys
sys.path.append("./") 
from fdtd3d import update_h, update_e
from fdtd3d import update_pbc_h, update_pbc_e

from scipy.constants import physical_constants
C = physical_constants['speed of light in vacuum'][0]


# setup
nx, ny, nz = n = 400, 400, 2
tmax, tgap = 1000, 20

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

# my pml with npml formula
npml, m = 100, 4
print (m+1.)/(15*np.pi*npml)
#sigma_dx = ( (np.linspace(1,npml,npml)/npml)**m )*(m+1.)/(15*np.pi*npml)
#sigma_dx = ( (np.linspace(1,npml,npml)/npml)**m )*0.1
sigma_dx = np.ones(npml, 'f')*(0.1)
pml_ca = (2 + sigma_dx[:])/(2 - sigma_dx[:])
pml_cb = sigma_dx[:]/(2 - sigma_dx[:])
print sigma_dx
print pml_ca
print pml_cb
psi_hy_x = np.zeros((npml,ny,nz), 'f')
psi_ez_x = np.zeros((npml,ny,nz), 'f')

# main loop
for tstep in xrange(1, tmax+1):
	update_e(*(em_arrays + ce_arrays))
	# for pml
	psi_ez_x[0,:-1,:] = pml_cb[0]*(hy[-npml,:-1,:] - hy[-npml-2,:-1,:])
	for i in xrange(1,npml):
		psi_ez_x[i,:-1,:] = pml_ca[i]*psi_ez_x[i-1,:-1,:] + pml_cb[i]*(hy[-npml+i,:-1,:] - hy[-npml-2+i,:-1,:])
	ez[-npml-1:-1,:-1,:] += 0.5*psi_ez_x[:,:-1,:]

	# for source
	ez[270,ny/2,1] += np.sin(2*np.pi*frequency*dt*tstep)
	# for pbc
	update_pbc_e('z', *em_arrays[:-3])


	update_h(*em_arrays)
	# for pml
	psi_hy_x[0,:,1:] = pml_cb[0]*(ez[-npml,:,1:] - ez[-npml-2,:,1:])
	for i in xrange(1,npml):
		psi_hy_x[i,:,1:] = pml_ca[i]*psi_hy_x[i-1,:,1:] + pml_cb[i]*(ez[-npml+i,:,1:] - ez[-npml-2+i,:,1:])
	hy[-npml:,:,1:] += 0.5*psi_hy_x[:,:,1:]
	# for pbc
	update_pbc_h('z', *em_arrays[3:])


	if tstep%tgap == 0:
		print_time(tstep)
		#print hx[260:280,190:210,1]
		print psi_ez_x[:,ny/2,nz/2]
		im.set_array(ez[:,:,nz/2].T**2)
		plt.draw()
		#savefig('./png/%.5d.png' % tstep) 

print ''
