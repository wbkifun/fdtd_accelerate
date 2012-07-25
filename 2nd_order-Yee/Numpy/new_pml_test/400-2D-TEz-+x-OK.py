#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


# setup
nx, ny = n = 400, 800
tmax, tgap = 1000, 10
npml = 100

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5
wavelength = 300e-9
frequency = C/wavelength

# allocate arrays
ez, hx, hy = [np.zeros(n, 'f') for i in range(3)]
cez = np.ones(n, 'f')*0.5

# prepare for plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8,12))
plt.ion()
im = plt.imshow(np.ones((nx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()

from matplotlib.patches import Rectangle
rect = Rectangle((nx-npml, 0), npml, ny, alpha=0.2)
plt.gca().add_patch(rect)

# prepare pml
sigma_dt = ( (np.linspace(1,npml,npml)/npml)**4 )*0.1
#sigma_dt = np.ones(npml, 'f')*(0.01)
pml_ca = (2. - sigma_dt[:])/(2 + sigma_dt[:])
pml_cb = 1./(2 + sigma_dt[:])

# main loop
for tstep in xrange(1, tmax+1):
	ez[:-npml-1,:-1] += cez[:-npml-1,:-1] * (hy[1:-npml,:-1] - hy[:-npml-1,:-1] - hx[:-npml-1,1:] + hx[:-npml-1,:-1])

	# for pml
	ez[-npml-1:-1,:-1] = pml_ca[:,np.newaxis] * ez[-npml-1:-1,:-1] + 2 * cez[-npml-1:-1,:-1] * pml_cb[:,np.newaxis] * (hy[-npml:,:-1] - hy[-npml-1:-1,:-1] - hx[-npml-1:-1,1:] + hx[-npml-1:-1,:-1])

	# for source
	ez[250,ny/2] += np.sin(2*np.pi*frequency*dt*tstep)
	#ez[250,ny/2] += np.exp( -0.5*( float(tstep-100)/20 )**2 )

	#hx[:-npml,1:] -= 0.5 * (ez[:-npml,1:] - ez[:-npml,:-1])
	hx[:,1:] -= 0.5 * (ez[:,1:] - ez[:,:-1])
	hy[1:-npml,:] -= 0.5 * (- ez[1:-npml,:] + ez[:-npml-1,:])

	# for pml
	hy[-npml:,:] = pml_ca[:,np.newaxis] * hy[-npml:,:] - pml_cb[:,np.newaxis] * (- ez[-npml:,:] + ez[-npml-1:-1,:])


	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()
		im.set_array(ez[:,:].T**2)
		plt.draw()

print ''
