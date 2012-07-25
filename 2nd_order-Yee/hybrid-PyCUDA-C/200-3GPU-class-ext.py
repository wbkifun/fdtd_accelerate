#!/usr/bin/env python

import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pycuda.driver as cuda
cuda.init()
ngpu = cuda.Device.count()

sys.path.append("./") 
from fdtd3d import *


nx, ny, nz = 512, 480, 480
tmax, tgap = 300, 10

print 'rank= %d, (%d, %d, %d)' % (rank, nx, ny, nz),
total_bytes = nx*ny*nz*np.nbytes['float32']*9
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )


fdtd = FDTD3DGPU(nx, ny, nz, rank, cuda)
fdtd.alloc_eh_fields()
fdtd.alloc_coeff_arrays()
fdtd.alloc_exchange_boundaries()
fdtd.prepare_functions()
if rank == 0: mpi_direction = 'b'
elif rank == 2: mpi_direction = 'f'
else: mpi_direction = 'fb'

if rank == 0:
	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((3*nx,ny),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
	colorbar()

	# measure kernel execution time
	from datetime import datetime
	t1 = datetime.now()
	flop = 3*(nx*ny*nz*30)*tgap
	flops = np.zeros(tmax/tgap+1)
	start, stop = cuda.Event(), cuda.Event()
	start.record()

# main loop
for tn in xrange(1, tmax+1):
	fdtd.update_h()
	fdtd.mpi_exchange_boundary_h(mpi_direction, comm)

	fdtd.update_e()
	fdtd.mpi_exchange_boundary_e(mpi_direction, comm)

	if rank == 1: fdtd.update_src(tn)

	if tn%tgap == 0 and rank == 0:
		stop.record()
		stop.synchronize()
		flops[tn/tgap] = flop/stop.time_since(start)*1e-6
		print '[',datetime.now()-t1,']'," %d/%d (%d %%) %1.2f GFLOPS\r" % (tn, tmax, float(tn)/tmax*100, flops[tn/tgap]),
		sys.stdout.flush()
		start.record()

if rank == 0: print "\navg: %1.2f GFLOPS" % flops[2:-2].mean()

g = np.zeros((nx,ny,nz),'f')
cuda.memcpy_dtoh(g, fdtd.ez_gpu)
if rank != 0:
	comm.Send(g, 0, 24)
else:
	lg = np.zeros((3*nx,ny),'f')
	lg[:nx,:] = g[:,:,nz/2]
	comm.Recv(g, 1, 24) 
	lg[nx:-nx,:] = g[:,:,nz/2]
	comm.Recv(g, 2, 24) 
	lg[2*nx:,:] = g[:,:,nz/2]
	imsh.set_array( lg.T**2 )
	show()#draw()
	#savefig('./png-wave/%.5d.png' % tstep) 


fdtd.finalize()
