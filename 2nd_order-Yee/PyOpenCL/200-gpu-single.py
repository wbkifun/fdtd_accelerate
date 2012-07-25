#!/usr/bin/env python

from fdtd3d import Fdtd3d
import numpy as np


nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
#nx, ny, nz = 256, 480, 960
tmax, tgap = 2000, 10

nxs = nx
s = Fdtd3d(nxs, ny, nz, target_device='gpu0')


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
	s.update_e()

	ez_gpu = s.eh_fieldss[0][2]
	s.programs[0].update_src(s.queues[0], (s.gsizes[0],), (s.lsize,), np.float32(tstep), ez_gpu)

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		cl.enqueue_read_buffer(s.queues[0], ez_gpu, f)
		imsh.set_array( f[:,:,nz/2].T**2 )
		plt.draw()

print('')
