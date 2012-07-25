#!/usr/bin/env python

from fdtd3d import Fdtd3d
import numpy as np


nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
#nx, ny, nz = 256, 480, 960
tmax, tgap = 200, 10

s = Fdtd3d(nx, ny, nz, target_device='cpu')


# Plot
import matplotlib.pyplot as plt
plt.ion()

imsh = plt.imshow(np.ones((nx, ny), 'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()


# Main loop
import sys
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	s.update_h()
	s.update_e()

	ez = s.eh_fieldss[0][2]
	ez[2*nx/3,ny/2,:] += np.sin(0.1*tstep)	# ez

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		imsh.set_array( ez[:,:,nz/2].T**2 )
		plt.draw()

print('')
