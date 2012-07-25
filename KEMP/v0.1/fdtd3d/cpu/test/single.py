#!/usr/bin/env python

import sys
sys.path.append('/home/kifang')

from kemp.fdtd3d.cpu import Fields, DirectSrc, GetFields
import numpy as np


nx, ny, nz = 240, 320, 320
tmax, tgap = 200, 10

fdtd = Fields(nx, ny, nz, coeff_use='', use_cpu_core=0)
src = DirectSrc(fdtd, 'ez', (nx/5*4, ny/2, 0), (nx/5*4, ny/2, nz-1), lambda tstep: np.sin(0.1 * tstep))
output = GetFields(fdtd, 'ez', (0, 0, nz/2), (nx-1, ny-1, nz/2))


# Plot
import matplotlib.pyplot as plt
plt.ion()
imag = plt.imshow(fdtd.ez[:,:,nz/2].T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.05)
plt.colorbar()


# Main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	fdtd.update_e()
	src.update(tstep)

	fdtd.update_h()

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		output.get_event().wait()
		imag.set_array(output.get_fields().T**2 )
		#plt.savefig('./single.png')
		plt.draw()

print('[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
'''
imag.set_array(fdtd.ez[:,:,nz/5*4].T**2 )
plt.savefig('./simple.png')
plt.show()
'''

print('')
