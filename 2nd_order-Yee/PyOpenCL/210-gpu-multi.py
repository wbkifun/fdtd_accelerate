#!/usr/bin/env python

from fdtd3d import Fdtd3d
import numpy as np


nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
#nx, ny, nz = 256, 480, 960
tmax, tgap = 200, 10

nxs = nx*3
s = Fdtd3d(nxs, ny, nz, target_device='gpu')


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
	#s.exchange_boundary_h()
	s.update_e()
	#s.exchange_boundary_e()
	ez_gpu = s.eh_fieldss[1][2]	# dev #1
	s.programs[1].update_src(s.queues[1], (s.gsizes[1],), (s.lsize,), np.float32(tstep), ez_gpu)

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		for i, (nx, queue) in enumerate(zip(s.nxs, s.queues)):
			f = np.zeros((nx, ny, nz), 'f')
			ez_gpu = s.eh_fieldss[i][2]
			cl.enqueue_read_buffer(queue, ez_gpu, f)
			global_f[i*nx:(i+1)*nx,:] = f[:,:,nz/2]
		imsh.set_array( global_f.T**2 )
		plt.draw()

print('')
