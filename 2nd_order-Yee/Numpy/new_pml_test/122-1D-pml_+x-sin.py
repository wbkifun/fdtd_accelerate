#!/usr/bin/env python

import numpy as np
import sys
sys.path.append("./") 
from fdtd3d import update_h, update_e
from fdtd3d import update_pbc_h, update_pbc_e

from scipy.constants import physical_constants
C = physical_constants['speed of light in vacuum'][0]


# setup
nx = n = 1000
tmax, tgap = 1000, 10

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5
wavelength = 30*dx
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

# my pml with npml formula
npml, m = 300, 4
print (m+1.)/(15*np.pi*npml)
#sigma_dx = ( (np.linspace(1,npml,npml)/npml)**m )*(m+1.)/(15*np.pi*npml)
#sigma_dx = ( (np.linspace(1,npml,npml)/npml)**m )*0.1
sigma_dx = np.ones(npml, 'f')*(0.01)
pml_ca = (2 - sigma_dx[:])/(2 + sigma_dx[:])
pml_cb = sigma_dx[:]/(2 + sigma_dx[:])
print sigma_dx
print pml_ca
print pml_cb
psi_hy_x = np.zeros((npml), 'f')
psi_ez_x = np.zeros((npml), 'f')

# main loop
for tstep in xrange(1, tmax+1):
	ez[:-1] += 0.5*(hy[1:] - hy[:-1])

	# for pml
	psi_ez_x[-1] = -pml_cb[-1]*(hy[-1] - hy[-3])
	for i in xrange(-2,-npml-1,-1):
		psi_ez_x[i] = pml_ca[i]*psi_ez_x[i+1] - pml_cb[i]*(hy[i] - hy[i-2])
	ez[-npml-2:-2] += 0.5*psi_ez_x[:]

	# for source
	ez[650] += np.sin(2*np.pi*frequency*dt*tstep)

	hy[1:] -= 0.5*(- ez[1:] + ez[:-1])

	# for pml
	psi_hy_x[-1] = -pml_cb[-1]*(ez[-1] - ez[-3])
	for i in xrange(-2,-npml-1,-1):
		psi_hy_x[i] = pml_ca[i]*psi_hy_x[i+1] - pml_cb[i]*(ez[i] - ez[i-2])
	hy[-npml-1:-1] += 0.5*psi_hy_x[:]


	if tstep%tgap == 0:
		line.set_ydata(ez[:])
		plt.draw()
