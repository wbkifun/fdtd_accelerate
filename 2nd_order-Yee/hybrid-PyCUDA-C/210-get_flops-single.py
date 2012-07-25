#!/usr/bin/env python

import sys

import pycuda.driver as cuda
cuda.init()
ngpu = cuda.Device.count()

sys.path.append("./") 
from fdtd3d import *
from datetime import datetime


def calc_flops_cpu(fdtd):
	tmax = 100
	flop = fdtd.nx*fdtd.ny*fdtd.nz*30
	flops = np.zeros(tmax+1)
	t1 = datetime.now()
	for tn in xrange(1, tmax+1):
		fdtd.update_h()
		fdtd.update_e()
		t2 = datetime.now()
		flops[tn] = flop/((t2-t1).seconds + (t2-t1).microseconds*1e-6)*1e-9
		t1 = datetime.now()
	return flops[2:-2].mean()


def calc_flops_gpu(fdtd):
	tmax = 100
	flop = fdtd.nx*fdtd.ny*fdtd.nz*30
	flops = np.zeros(tmax+1)
	start, stop = cuda.Event(), cuda.Event()
	start.record()
	for tn in xrange(1, tmax+1):
		fdtd.update_h()
		fdtd.update_e()
		stop.record()
		stop.synchronize()
		flops[tn] = flop/stop.time_since(start)*1e-6
		start.record()
	return flops[2:-2].mean()


nx, ny, nz = 128, 480, 480	# seed
print 'Calculation FLOPS for single device.'
print 'Seed: (%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx*ny*nz*np.nbytes['float32']*9
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

flops = np.zeros(4)
fdtd_list = [FDTD3DCPU(nx, ny, nz), FDTD3DGPU(nx, ny, nz, 0, cuda),
		FDTD3DGPU(nx, ny, nz, 0, cuda), FDTD3DGPU(nx, ny, nz, 0, cuda)]

for i, fdtd in enumerate( fdtd_list ):
	fdtd.alloc_eh_fields()
	fdtd.alloc_coeff_arrays()
	fdtd.alloc_exchange_boundaries()
	fdtd.prepare_functions()
	if i == 0: flops[i] = calc_flops_cpu(fdtd)
	else: flops[i] = calc_flops_gpu(fdtd)
	fdtd.finalize()

print 'GFLOPS: ', flops
print 'Total: %1.2f GFLOPS' % ( flops.sum() )
