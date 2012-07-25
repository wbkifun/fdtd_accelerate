#!/usr/bin/env python

import numpy as np
import sys
from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0


# setup
nx = n = 400
tmax, tgap = 2000, 20
npml = 100

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5
wavelength = 30*dx
frequency = C/wavelength

# allocate arrays
ez = np.zeros(n, 'f')
hy = np.zeros(n, 'f')

# prepare for plot
import matplotlib.pyplot as plt
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
sigma_dt = ( (np.linspace(1,npml,npml+1)/npml)**m )*0.1
#sigma_dt = np.ones(npml+1, 'f')*(0.001)
pml_ca = (2 - sigma_dt[:])/(2 + sigma_dt[:])
pml_cb = sigma_dt[:]/(2 + sigma_dt[:])
psi_hy_x = np.zeros((npml+1), 'f')
psi_ez_x = np.zeros((npml+1), 'f')

# main loop
for tstep in xrange(1, tmax+1):
	ez[:-1] += 0.5*(hy[1:] - hy[:-1])

	# for pml
	ez[-npml-1:-1] += 0.5*(psi_ez_x[1:] - psi_ez_x[:-1])
	psi_hy_x[:-1] -= pml_cb[:-1]*ez[-npml-1:-1]
	psi_ez_x[1:] = pml_ca[1:]*psi_ez_x[1:] - pml_cb[1:]*hy[-npml:]

	# for source
	#ez[650] += np.sin(2*npml.pi*frequency*dt*tstep)
	ez[250] += np.exp( -0.5*( float(tstep-100)/20 )**2 )

	hy[1:] -= 0.5*(- ez[1:] + ez[:-1])

	# for pml
	hy[-npml:] += 0.5*(psi_hy_x[1:] - psi_hy_x[:-1])
	psi_ez_x[1:] -= pml_cb[1:]*hy[-npml:]
	psi_hy_x[:-1] = pml_ca[:-1]*psi_hy_x[:-1] - pml_cb[:-1]*ez[-npml-1:-1]


	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()
		line.set_ydata(ez[:])
		plt.draw()

print ''
