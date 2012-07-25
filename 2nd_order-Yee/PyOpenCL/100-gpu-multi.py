#!/usr/bin/env python

from fdtd3d_gpu_cpu import Fdtd3dGpu
import numpy as np


nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
#nx, ny, nz = 256, 480, 960
tmax, tgap = 200, 10

nxs = nx*3
s = Fdtd3dGpu(nxs, ny, nz, target_device='all')


# Plot
import matplotlib.pyplot as plt
plt.ion()

global_f = np.ones((nxs, ny), 'f')
imsh = plt.imshow(global_f.T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()


# Main loop
import sys
import pyopencl as cl
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	s.update_h()
	s.exchange_boundary_h()
	s.update_e()
	s.exchange_boundary_e()
	s.programs[1].update_src(s.queues[1], (s.Gs,), (s.Ls,), np.float32(tstep), s.eh_fields_gpus[1][2])	# dev #1, ez_gpu

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		for i, (nx, queue) in enumerate(zip(s.nxs, s.queues)):
			f = np.zeros((nx, ny, nz), 'f')
			cl.enqueue_read_buffer(queue, s.eh_fields_gpus[i][2], f)	# ez_gpu
			global_f[i*nx:(i+1)*nx,:] = f[:,:,nz/2]
		imsh.set_array( global_f.T**2 )
		plt.draw()

print('')
