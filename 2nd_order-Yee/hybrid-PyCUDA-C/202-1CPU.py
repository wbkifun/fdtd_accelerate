#!/usr/bin/env python

import sys

sys.path.append("./") 
from fdtd3d import *


tmax, tgap = 100, 10
nx, ny, nz = 128, 480, 480

print '(%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx*ny*nz*np.nbytes['float32']*9
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )


fdtd = FDTD3DCPU(nx, ny, nz)
fdtd.alloc_eh_fields()
fdtd.alloc_coeff_arrays()
fdtd.alloc_exchange_boundaries()
fdtd.prepare_functions()

# prepare for plot
from matplotlib.pyplot import *
ion()
imsh = imshow(np.ones((nx,ny),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
colorbar()

# measure kernel execution time
from datetime import datetime
flop = nx*ny*nz*30*tgap
flops = np.zeros(tmax/tgap+1)
t0 = datetime.now()
t1 = datetime.now()

# main loop
for tn in xrange(1, tmax+1):
	fdtd.update_h()
	fdtd.update_e()
	fdtd.update_src(tn)

	if tn%tgap == 0:
		t2 = datetime.now()
		flops[tn/tgap] = flop/((t2-t1).seconds + (t2-t1).microseconds*1e-6)*1e-9
		print "[%s] %d/%d (%d %%) %1.3f GFLOPS\r" % (t2-t0, tn, tmax, float(tn)/tmax*100, flops[tn/tgap]),
		sys.stdout.flush()
		t1 = datetime.now()

print "\navg: %1.2f GFLOPS" % flops[2:-2].mean()

imsh.set_array( fdtd.ez[:,:,nz/2].T**2 )
show()#draw()
#savefig('./png-wave/%.5d.png' % tstep) 

fdtd.finalize()
