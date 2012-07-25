#!/usr/bin/env python

import sys
sys.path.append('/home/kifang')

from kemp.fdtd3d import common_gpu
from kemp.fdtd3d.gpu import Fields, DirectSrc, GetFields, ExchangeFields
import numpy as np
import pyopencl as cl


nx, ny, nz = 240, 640, 640
tmax, tgap = 200, 10
divide_axes = 'x'

gpu_devices = common_gpu.get_gpu_devices()
context = cl.Context(gpu_devices)
ngpu = len(gpu_devices)

fdtds = [Fields(context, device, nx, ny, nz, coeff_use='') for device in gpu_devices]
outputs = [GetFields(fdtds[0], 'ez', (0, 0, nz/2), (nx-1, ny-1, nz/2))]
outputs += [GetFields(fdtd, 'ez', (1, 0, nz/2), (nx-1, ny-1, nz/2)) for fdtd in fdtds[1:]]
src = DirectSrc(fdtds[1], 'ez', (nx/5*4, ny/2, 0), (nx/5*4, ny/2, nz-1), lambda tstep: np.sin(0.1 * tstep))
exch = ExchangeFields(fdtds, 'x')


# Plot
import matplotlib.pyplot as plt
plt.ion()
global_ez = np.ones((ngpu*(nx-1) + 1, ny), dtype=fdtds[0].dtype)
imag = plt.imshow(global_ez.T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.05)
plt.colorbar()


# Main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	for fdtd in fdtds:
		fdtd.update_e()
	src.update(tstep)
	exch.update_e()

	for fdtd in fdtds:
		fdtd.update_h()
	exch.update_h()

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		idxs = [0] + [i*nx - (i-1) for i in range(1, ngpu+1)]
		for i, output in enumerate(outputs):
			output.get_event().wait()
			outf = output.get_fields()
			global_ez[idxs[i]:idxs[i+1],:] = output.get_fields('ez')

		imag.set_array(global_ez.T**2 )
		#plt.savefig('./simple.png')
		plt.draw()

#outputs[0].get_event().wait()
#print('[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
