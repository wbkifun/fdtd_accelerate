#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


# setup
nx = n = 400
tmax, tgap = 1000, 10
npml = 100

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5
wavelength = 300e-9
frequency = C/wavelength

# allocate arrays
ez = np.zeros(n, 'f')
hy = np.zeros(n, 'f')

# prepare for plot
import matplotlib.pyplot as plt
#plt.figure(figsize=(20,7))
plt.ion()
x = np.arange(n)
line, = plt.plot(x, np.ones(n, 'f'))
plt.axis([0,n,-1.5,1.5])

from matplotlib.patches import Rectangle
rect = Rectangle((nx-npml, -1.5), npml, 3, alpha=0.1)
plt.gca().add_patch(rect)

# prepare pml
m = 4
print (m+1.)/(15*np.pi*npml)*0.5/C
#sigma_dt = ( (np.linspace(1,npml,npml)/npml)**m )*(m+1.)/(15*np.pi*npml)*0.5/C
sigma_dt = ( (np.linspace(1,npml,npml)/npml)**m )*0.1
#sigma_dt = np.ones(npml, 'f')*0.01
pml_ca = (2. - sigma_dt[:])/(2 + sigma_dt[:])
pml_cb = 1./(2 + sigma_dt[:])

# main loop
for tstep in xrange(1, tmax+1):
	ez[:-npml-1] += 0.5 * (hy[1:-npml] - hy[:-npml-1])

	# for pml
	ez[-npml-1:-1] = pml_ca[:] * ez[-npml-1:-1] + pml_cb[:] * (hy[-npml:] - hy[-npml-1:-1])

	# for source
	#ez[650] += np.sin(2*np.pi*frequency*dt*tstep)
	ez[250] += np.exp( -0.5*( float(tstep-100)/20 )**2 )

	hy[1:-npml] -= 0.5 * (- ez[1:-npml] + ez[:-npml-1])

	# for pml
	hy[-npml:] = pml_ca[:] * hy[-npml:] - pml_cb[:] * (- ez[-npml:] + ez[-npml-1:-1]) 


	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()
		line.set_ydata(ez[:])
		plt.draw()

print ''
