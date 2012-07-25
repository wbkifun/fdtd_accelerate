#!/usr/bin/env python

import numpy as np
import sys
from scipy.constants import physical_constants
C = physical_constants['speed of light in vacuum'][0]

nz = n = 200
tmax, tgap = 100, 1

dx = 10e-9		# m
dt = 0.5*dx/C	# Courant factor S=0.5

wavelength = 30*dx
frequency = C/wavelength


# allocate arrays
#ex, hy = em_arrays = [np.zeros(n, 'f')]*2
ex, hy = em_arrays = [np.zeros(n, 'f') for i in range(2)]
cex = np.ones(n, 'f')*0.5

# prepare for plot
'''
import matplotlib.pyplot as plt
plt.ion()
imsh = plt.imshow(np.ones((nx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.05)
plt.colorbar()
'''
from matplotlib.pyplot import *
ion()
x = np.arange(nz)
line, = plot(x, ex)
ylim(-1.2, 1.2)

# measure execution time
from datetime import datetime
t0 = datetime.now()

print 2*np.pi*frequency*dt
# main loop
for tn in xrange(1, tmax+1):
	hy[1:] -= 0.5*(ex[1:] - ex[:-1])

	ex[:-1] += cex[:-1]*(- hy[1:] + hy[:-1])

	ex[nz/2] += np.sin(2*np.pi*frequency*dt*tn)
	print np.sin(2*np.pi*frequency*dt*tn)

	if tn%tgap == 0:
		t1 = datetime.now()
		#print "[%s] %d/%d(%d %%)\r" % (t1-t0, tn, tmax, float(tn)/tmax*100), 
		#sys.stdout.flush()

		line.set_ydata(ex[:])
		draw()
		#savefig('./png/%.5d.png' % tstep) 

print ''
