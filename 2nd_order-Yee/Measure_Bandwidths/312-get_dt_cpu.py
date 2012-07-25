#!/usr/bin/env python

from fdtd3d import Fdtd3d
from datetime import datetime
import sys

nx = int(sys.argv[1])
tmax = int(sys.argv[2])
ny, nz = nx, nx

s = Fdtd3d(nx, ny, nz, target_device='cpu', print_verbose=False)
ez = s.eh_fieldss[0][2]
#ez[:] = np.random.rand(nx,nx,nx).astype(np.float32)
#print ez.shape

t0 = datetime.now()
for tstep in xrange(1, tmax+1):
	s.update_h()
	s.update_e()
	#ez[2*nx/3,ny/2,:] += np.sin(0.1*tstep)

dt0 = datetime.now() - t0
dt = dt0.seconds + dt0.microseconds * 1e-6

print dt
