#!/usr/bin/env python

import numpy as np
import sys

nx = 500
tmax, tgap = 1000, 20

dx = 10.	# nm
dt = 0.5*dx	# Courant factor S=0.5
wavelength = 30*dx
frequency = 1/wavelength

ez = np.zeros(nx, dtype=np.float32)
hy = np.zeros(nx, dtype=np.float32)
cez = np.ones_like(ez)*0.5

# prepare for plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
line, = ax.plot(np.arange(nx), ez)
ax.set_ylim(-1.2, 1.2)

# measure execution time
from datetime import datetime
t0 = datetime.now()


for tn in xrange(1, tmax+1):
	#ez[nx/2] += np.sin(2*np.pi*frequency*dt*tn)
	ez[nx-200] += np.exp( -0.5*( float(tn-100)/20 )**2 )

	hy[1:] += 0.5*(ez[1:] - ez[:-1])

	ez[:-1] += cez[:-1]*(hy[1:] - hy[:-1])

	if tn%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d(%d %%)\r" % (t1-t0, tn, tmax, float(tn)/tmax*100), 
		sys.stdout.flush()

		line.set_ydata(ez)
		line.recache()
		plt.draw()

print ''
