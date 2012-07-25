#!/usr/bin/env python

import numpy as np
import sys

nx = 500
tmax, tgap = 1000, 20

dx = 10.	# nm
dt = 0.5*dx	# Courant factor S=0.5

ez = np.zeros(nx, dtype=np.float32)
hy = np.zeros(nx, dtype=np.float32)
cez = np.ones_like(ez)*0.5

# NPML
npml = 100
sigma = 0.002
pca = (2 - sigma*dt) / (2 + sigma*dt)
pcb = sigma*dt / (2 + sigma*dt)
psi_ez = np.zeros(npml+1, dtype=np.float32)
psi_hy = np.zeros_like(psi_ez)
print('pca=%g, pcb=%g' %(pca, pcb))


# prepare for plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
line, = ax.plot(np.arange(nx), ez)
ax.set_ylim(-1.2, 1.2)

from matplotlib.patches import Rectangle
rect = Rectangle((nx-npml, -1.2), npml, 2.4, alpha=0.1)
plt.gca().add_patch(rect)

# measure execution time
from datetime import datetime
t0 = datetime.now()


for tn in xrange(1, tmax+1):
	hy[1:] += 0.5*(ez[1:] - ez[:-1])

	# npml
	hy[-npml:] += 0.5*(psi_ez[1:] - psi_ez[:-1])
	psi_hy[:] -= pcb*hy[-npml-1:]
	psi_ez[:] = pca*psi_ez[:] - pcb*ez[-npml-1:]

	ez[:-1] += cez[:-1]*(hy[1:] - hy[:-1])

	# source
	ez[nx-200] += np.exp( -0.5*( float(tn-100)/20 )**2 )

	# npml
	ez[-npml-1:-1] += cez[-npml-1:-1]*(psi_hy[1:] - psi_hy[:-1])
	psi_ez[:] -= pcb*ez[-npml-1:]
	psi_hy[:] = pca*psi_hy[:] - pcb*hy[-npml-1:]


	if tn%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d(%d %%)\r" % (t1-t0, tn, tmax, float(tn)/tmax*100), 
		sys.stdout.flush()

		line.set_ydata(ez)
		line.recache()
		plt.draw()

print ''
