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


tmax, tgap = 200, 10
ny, nz = 480, 480
if rank == 0: nx = 40
else: nx = 512

print 'rank= %d, (%d, %d, %d)' % (rank, nx, ny, nz),
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
	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((nx+3*512,ny),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
	colorbar()

	# measure kernel execution time
	from datetime import datetime
	flop = (nx+3*512)*ny*nz*30*tgap
	flops = np.zeros(tmax/tgap+1)
	t0 = datetime.now()
	t1 = datetime.now()

# main loop
for tn in xrange(1, tmax+1):
	fdtd.update_h()
	fdtd.mpi_exchange_boundary_h(mpi_direction, comm)

	fdtd.update_e()
	fdtd.mpi_exchange_boundary_e(mpi_direction, comm)

	if rank == 1: fdtd.update_src(tn)

	if tn%tgap == 0 and rank == 0:
		t2 = datetime.now()
		flops[tn/tgap] = flop/((t2-t1).seconds + (t2-t1).microseconds*1e-6)*1e-9
		print "[%s] %d/%d (%d %%) %1.3f GFLOPS\r" % (t2-t0, tn, tmax, float(tn)/tmax*100, flops[tn/tgap]),
		sys.stdout.flush()
		t1 = datetime.now()

if rank == 0: print "\navg: %1.2f GFLOPS" % flops[2:-2].mean()

if rank != 0:
	g = np.zeros((nx,ny,nz),'f')
	cuda.memcpy_dtoh(g, fdtd.ez_gpu)
	comm.Send(g, 0, 24)
else:
	g = np.zeros((512,ny,nz),'f')
	lg = np.zeros((nx+3*512,ny),'f')
	lg[:nx,:] = fdtd.ez[:,:,nz/2]
	comm.Recv(g, 1, 24) 
	lg[nx:nx+512,:] = g[:,:,nz/2]
	comm.Recv(g, 2, 24) 
	lg[nx+512:nx+2*512,:] = g[:,:,nz/2]
	comm.Recv(g, 3, 24) 
	lg[nx+2*512:,:] = g[:,:,nz/2]
	imsh.set_array( lg.T**2 )
	show()#draw()
	#savefig('./png-wave/%.5d.png' % tstep) 


fdtd.finalize()
