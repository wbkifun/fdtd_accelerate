#!/usr/bin/env python

import sys
from datetime import datetime

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pycuda.driver as cuda
cuda.init()
ngpu = cuda.Device.count()

sys.path.append("./") 
from fdtd3d import *


ny, nz = 480, 480
if rank == 0: nx = 42
else: nx = 512
tmax = 100

if rank == 0:
	print 'Calculation FLOPS for multi GPUs and CPU.'
	print 'CPU: (%d, %d, %d)' % (nx, ny, nz),
	total_bytes = nx*ny*nz*np.nbytes['float32']*9
	if total_bytes/(1024**3) == 0:
		print '%d MB' % ( total_bytes/(1024**2) )
	else:
		print '%1.2f GB' % ( float(total_bytes)/(1024**3) )
elif rank == 1:
	print 'GPU %d: ' % ngpu,
	print '(%d, %d, %d)' % (nx, ny, nz),
	total_bytes = nx*ny*nz*np.nbytes['float32']*9
	if total_bytes/(1024**3) == 0:
		print '%d MB' % ( total_bytes/(1024**2) )
	else:
		print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

if rank == 0: fdtd = FDTD3DCPU(nx, ny, nz)
else: fdtd = FDTD3DGPU(nx, ny, nz, rank-1, cuda)
fdtd.alloc_eh_fields()
fdtd.alloc_coeff_arrays()
fdtd.alloc_exchange_boundaries()
fdtd.prepare_functions()

if rank == 0: mpi_direction = 'b'
elif rank == size-1: mpi_direction = 'f'
else: mpi_direction = 'fb'

if rank == 0:
	flop = (nx + ngpu*512)*ny*nz*30
	flops = np.zeros(tmax+1)
	t1 = datetime.now()

for tn in xrange(1, tmax+1):
	fdtd.update_h()
	fdtd.mpi_exchange_boundary_h(mpi_direction, comm)
	fdtd.update_e()
	fdtd.mpi_exchange_boundary_e(mpi_direction, comm)

	if rank == 0:
		t2 = datetime.now()
		flops[tn] = flop/((t2-t1).seconds + (t2-t1).microseconds*1e-6)*1e-9
		t1 = datetime.now()

if rank == 0: print 'Total: %1.2f GFLOPS' % ( flops[2:-2].mean() )
fdtd.finalize()
