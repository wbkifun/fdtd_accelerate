#!/usr/bin/env python

import sys
import numpy as np


# Parameter setup
nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
#nx, ny, nz = 256, 480, 960
tmax, tgap = 200, 10

print '(%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx*ny*nz*4*9
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

if nz%4 != 0:
	print "Error: nz is not multiple of 4"
	sys.exit()


# Allocation
eh_fields = ex, ey, ez, hx, hy, hz = [np.zeros((nx,ny,nz), dtype=np.float32) for i in xrange(6)]
ce_fields = cex, cey, cez = [np.ones((nx,ny,nz), dtype=np.float32)*0.5 for i in xrange(3)]


# Program and Kernel
import subprocess
kkeys = ['NXYZ', 'NYZ', 'NXY', 'NX', 'NY', 'NZ', 'OMP_MAX_THREADS']
kvals = [str(nx*ny*nz), str(ny*nz), str(nx*ny), str(nx), str(ny), str(nz), str(4)]

kernels = open('fdtd3d.c').read()
for key, val in zip(kkeys, kvals):
	kernels = kernels.replace(key, val)

of = open('/tmp/fdtd3d.c', 'w')
of.write(kernels)
of.close()
cmd = 'gcc -O3 -std=c99 -fpic -shared -fopenmp -msse %s -o /tmp/libfdtd3d.so' %(of.name)
subprocess.Popen(cmd.split())
clib = np.ctypeslib.load_library('libfdtd3d', '/tmp/')
arg = np.ctypeslib.ndpointer(dtype=ex.dtype, ndim=ex.ndim, shape=ex.shape, flags='C_CONTIGUOUS, ALIGNED')
clib.update_h.argtypes = [arg for i in xrange(6)]
clib.update_e.argtypes = [arg for i in xrange(9)]
clib.update_h.restype = None
clib.update_e.restype = None


# Plot
import matplotlib.pyplot as plt
plt.ion()
imsh = plt.imshow(np.ones((nx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()


# Main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	clib.update_h(*eh_fields)
	clib.update_e(*(eh_fields+ce_fields))
	ez[nx/2,ny/2,:] += np.sin(0.1*tstep)

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		imsh.set_array( ez[:,:,nz/2].T**2 )
		plt.draw()

print('')
