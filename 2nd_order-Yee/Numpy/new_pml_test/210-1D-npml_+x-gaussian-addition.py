#!/usr/bin/env python

import numpy as np
import sys
from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0


# setup
nx = n = 1000
tmax, tgap = 2000, 10
npml = 300

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5
wavelength = 300e-9
frequency = C/wavelength

# allocate arrays
ez = np.zeros(n, 'f')
hy = np.zeros(n, 'f')

# prepare for plot
import matplotlib.pyplot as plt
plt.figure(figsize=(20,7))
plt.ion()
x = np.arange(n)
line, = plt.plot(x, np.ones(n, 'f'))
plt.axis([0,n,-1.5,1.5])

from matplotlib.patches import Rectangle
rect = Rectangle((nx-npml, -1.5), npml, 3, alpha=0.1)
plt.gca().add_patch(rect)

# my pml with npml formula
m = 4
print (m+1.)/(15*np.pi*npml)*0.5/C
#sigma_dt = ( (np.linspace(1,npml,npml+1)/npml)**m )*(m+1.)/(15*np.pi*npml)*0.5/C
#sigma_dt = ( (np.linspace(1,npml,npml+1)/npml)**m )*0.002
sigma_dt = np.ones(npml+1, 'f')*(0.002)
psi_ez = np.zeros(npml+1, 'f')
psi_hy = np.zeros(npml+1, 'f')

# main loop
for tstep in xrange(1, tmax+1):
	psi_ez[:] -= sigma_dt[:]*ez[-npml-2:-1]
	ez[:-1] += 0.5*(hy[1:] - hy[:-1])
	ez[-npml-1:-1] += 0.5*(psi_hy[1:] - psi_hy[:-1])
	psi_ez[:] -= sigma_dt[:]*ez[-npml-2:-1]

	# for source
	#ez[650] += np.sin(2*np.pi*frequency*dt*tstep)
	ez[650] += np.exp( -0.5*( float(tstep-100)/20 )**2 )

	psi_hy[:] -= sigma_dt[:]*hy[-npml-1:]
	hy[1:] -= 0.5*(- ez[1:] + ez[:-1])
	hy[-npml:] -= 0.5*(- psi_ez[1:] + psi_ez[:-1])
	psi_hy[:] -= sigma_dt[:]*hy[-npml-1:]


	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()
		line.set_ydata(ez[:])
		plt.draw()

print ''
