#!/usr/bin/env python

from fdtd3d_gpu_cpu import Fdtd3dGpu
import numpy as np


nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
#nx, ny, nz = 256, 480, 960
tmax, tgap = 200, 10

s = Fdtd3dGpu(nx, ny, nz, target_device=0)


# Plot
import matplotlib.pyplot as plt
plt.ion()

f = np.ones((nx, ny, nz), 'f')
imsh = plt.imshow(f[:,:,nz/2].T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
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
	s.programs[0].update_src(s.queues[0], (s.Gs,), (s.Ls,), np.float32(tstep), s.eh_fields_gpus[0][2])	# dev #1, ez_gpu

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		cl.enqueue_read_buffer(s.queues[0], s.eh_fields_gpus[0][2], f)	# ez_gpu
		imsh.set_array( f[:,:,nz/2].T**2 )
		plt.draw()

print('')
