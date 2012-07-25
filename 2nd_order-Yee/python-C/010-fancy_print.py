#!/usr/bin/env python

import numpy as np
import sys
import dielectric

def set_c(f, direction):
	f[:,:,:] = 0.5
	if 'x' in direction: f[-1,:,:] = 0
	if 'y' in direction: f[:,-1,:] = 0
	if 'z' in direction: f[:,:,-1] = 0

	return f


nx, ny, nz = 240, 256, 256
tmax, tgap = 100, 10

print '(%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx*ny*nz*4*9
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

# memory allocate
f = np.zeros((nx,ny,nz),'f')
#f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz))

ex = f.copy()
ey = f.copy()
ez = f.copy()
hx = f.copy()
hy = f.copy()
hz = f.copy()

cex = set_c(np.zeros_like(f),'yz')
cey = set_c(np.zeros_like(f),'zx')
cez = set_c(np.zeros_like(f),'xy')

eh_fields = [ex, ey, ez, hx, hy, hz]
ce_fields = [cex, cey, cez]

'''
# prepare for plot
from matplotlib.pyplot import *
ion()
imsh = imshow(np.ones((nx,ny),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
colorbar()
'''

# measure kernel execution time
from datetime import datetime
flop = (nx*ny*nz*30)*tgap
flops = np.zeros(tmax/tgap+1)
t1 = datetime.now()

# main loop
for tn in xrange(1, tmax+1):
	dielectric.update_h(*eh_fields)
	dielectric.update_e(*(eh_fields+ce_fields))
	ez[nx/2,ny/2,:] += np.sin(0.1*tn)

	if tn%tgap == 0:
		dt = datetime.now()-t1
		flops[tn/tgap] = flop/(dt.seconds + dt.microseconds*1e-6)*1e-9
		print "[%s] %d/%d (%d %%) %1.3f GFLOPS\r" % (dt, tn, tmax, float(tn)/tmax*100, flops[tn/tgap]),
		sys.stdout.flush()
		#imsh.set_array( ez[:,:,nz/2].T**2 )
		#draw()
		#savefig('./png-wave/%.5d.png' % tn) 
		t1 = datetime.now()

print "\navg: %1.2f GFLOPS" % flops[2:-2].mean()

