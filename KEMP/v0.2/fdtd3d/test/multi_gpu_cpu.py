#!/usr/bin/env python

import sys
sys.path.append('/home/kifang')

from kemp.fdtd3d import common_gpu, common_cpu
from kemp.fdtd3d import gpu, cpu
from kemp.fdtd3d.exchange_boundary import ExchangeInternal
import numpy as np
import pyopencl as cl


nx_gpu = 120
nx_cpu = nx_gpu/5
ny, nz = 320, 320
tmax, tgap = 2000, 5
divide_axes = 'x'

# GPUs
gpu_devices = common_gpu.get_gpu_devices()
context = cl.Context(gpu_devices)
ngpu = len(gpu_devices)

fdtds = [gpu.Fields(context, device, nx_gpu, ny, nz, coeff_use='') for device in gpu_devices]
outputs = [gpu.GetFields(fdtd, 'ez', (0, 0, nz/2), (nx_gpu-2, ny-1, nz/2)) for fdtd in fdtds]
src_e = gpu.DirectSrc(fdtds[2], 'ez', (nx_gpu/5*1, ny/2, 0), (nx_gpu/5*1, ny/2, nz-1), lambda tstep: np.sin(0.1 * tstep))


# CPU
common_cpu.print_cpu_info()
fdtds.append( cpu.Fields(nx_cpu, ny, nz, coeff_use='') )
outputs.append( cpu.GetFields(fdtds[-1], 'ez', (0, 0, nz/2), (nx_cpu-2, ny-1, nz/2)) )


# GPUs-CPU
exch = ExchangeInternal(fdtds, 'x')


# Plot
import matplotlib.pyplot as plt
plt.ion()
idxs = [0] + [i*nx_gpu - i for i in range(1, ngpu+1)] + [ngpu*nx_gpu + nx_cpu - (ngpu+1)]
for idx in idxs[1:]:
	plt.plot((idx,idx), (0,ny), color='w', linewidth=0.2)

global_ez = np.ones((idxs[-1], ny), dtype=fdtds[0].dtype)
imag = plt.imshow(global_ez.T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.05)
plt.colorbar()



# Main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	for fdtd in fdtds:
		fdtd.update_e()
	src_e.update(tstep)
	exch.update_e()

	for fdtd in fdtds:
		fdtd.update_h()
	exch.update_h()

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

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
