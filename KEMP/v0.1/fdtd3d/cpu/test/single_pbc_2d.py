#!/usr/bin/env python

import sys
sys.path.append('/home/kifang')

from kemp.fdtd3d import common_gpu
from kemp.fdtd3d.gpu import Fields, DirectSrc, GetFields, PbcInt
import numpy as np
import pyopencl as cl


nx, ny, nz = 2, 640, 640
tmax, tgap = 1000, 10
gpu_id = 0

gpu_devices = common_gpu.get_gpu_devices()
context = cl.Context(gpu_devices)
device = gpu_devices[gpu_id]

fdtd = Fields(context, device, nx, ny, nz, coeff_use='')
src = DirectSrc(fdtd, 'ex', (1, ny/5*4, nz/5*3), (1, ny/5*4, nz/5*3), lambda tstep: np.sin(0.1 * tstep))
pbc = PbcInt(fdtd, 'x')
output = GetFields(fdtd, 'ex', (1, 0, 0), (1, ny-1, nz-1))


# Plot
import matplotlib.pyplot as plt
plt.ion()
imag = plt.imshow(output.get_fields().T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.05)
plt.colorbar()


# Main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	fdtd.update_e()
	src.update(tstep)
	pbc.update_e()

	fdtd.update_h()
	pbc.update_h()

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		output.get_event().wait()
		f = output.get_fields()
		imag.set_array(f.T**2 )
		#plt.savefig('./simple.png')
		plt.draw()

#output.get_event().wait()
#print('[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
