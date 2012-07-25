#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


# setup
nx = n = 100
tmax, tgap = 800, 5
npml = 10

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5
wavelength = 300e-9
frequency = C/wavelength

# allocate arrays
ez1 = np.zeros(n, 'f')
hy1 = np.zeros(n, 'f')
ez2 = np.zeros(n, 'f')
hy2 = np.zeros(n, 'f')

# prepare for plot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
rect1 = Rectangle((nx-npml, -1.2), npml, 2.4, alpha=0.1)
rect2 = Rectangle((nx-npml, -1.2), npml, 2.4, alpha=0.1)
plt.ion()
fig = plt.figure()	#figsize=(20,7))
x = np.arange(n)

ax1 = fig.add_subplot(2,1,1)
line1, = ax1.plot(x, np.ones(nx, 'f'))
ax1.set_ylim(-1.2, 1.2)
ax1.add_patch(rect1)

ax2 = fig.add_subplot(2,1,2)
line2, = ax2.plot(x, np.ones(nx, 'f'))
ax2.set_ylim(-1.2, 1.2)
ax2.add_patch(rect2)

# prepare pml
#s1 = np.ones(npml, 'f') * 0.01
s1 = ( (np.linspace(1,npml,npml) / npml) ** 4 ) * 1.	# sigma * dt * dx^4
s2 = ( (np.linspace(1,npml,npml) / npml) ** 4 ) * 0.1
pca1 = (2. - s1[:]) / (2 + s1[:])	# pml coefficients
pcb1 = 1. / (2 + s1[:])
pca2 = (2. - s2[:]) / (2 + s2[:]) 
pcb2 = (1. + 4 * s2[:]) / (2 + s2[:])
pcc2 = (1. - 4 * s2[:]) / (1. + 4 * s2[:])

# main loop
for tstep in xrange(1, tmax+1):
	#pulse = np.sin(2*np.pi*frequency*dt*tstep)
	pulse = np.exp( -0.5*( float(tstep-100)/10 )**2 )

	# polynomial sigma
	ez1[:-npml-1] += 0.5 * (hy1[1:-npml] - hy1[:-npml-1])
	ez1[-npml-1:-1] = pca1[:] * ez1[-npml-1:-1] + pcb1[:] * (hy1[-npml:] - hy1[-npml-1:-1])
	ez1[60] += pulse
	hy1[1:-npml] -= 0.5 * (- ez1[1:-npml] + ez1[:-npml-1])
	hy1[-npml:] = pca1[:] * hy1[-npml:] - pcb1[:] * (- ez1[-npml:] + ez1[-npml-1:-1]) 

	# polynomial sigma with correction term
	ez2[:-npml-1] += 0.5 * (hy2[1:-npml] - hy2[:-npml-1])
	ez2[-npml-1:-1] = pca2[:] * ez2[-npml-1:-1] + pcb2[:] * (hy2[-npml:] - hy2[-npml-1:-1])
	ez2[60] += pulse
	hy2[1:-npml] -= 0.5 * (- ez2[1:-npml] + ez2[:-npml-1])
	hy2[-npml:] = pca2[:] * hy2[-npml:] - pcb2[:] * (- ez2[-npml:] + pcc2[:] * ez2[-npml-1:-1]) 


	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()
		line1.set_ydata(ez1[:])
		line2.set_ydata(ez2[:])
		plt.draw()

print ''
