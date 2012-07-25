#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cuda.init()
ngpu = cuda.Device.count()


class Fdtd3DGpu:
	def __init__(s, nx, ny, nz):
		s.nx, s.ny, s.nz = nx, ny, nz
		s.Dx, s.Dy = 32, 16
		s.rank = comm.Get_rank()

		if s.nz%s.Dx != 0:
			print "Error: nz is not multiple of %d" % (s.Dx)
			sys.exit()
		if (s.nx*s.ny)%s.Dy != 0:
			print "Error: nx*ny is not multiple of %d" % (s.Dy)
			sys.exit()
		print 'rank= %d, (%d, %d, %d)' % (s.rank, s.nx, s.ny, s.nz),
		total_bytes = s.nx*s.ny*s.nz*np.nbytes['float32']*9
		if total_bytes/(1024**3) == 0:
			print '%d MB' % ( total_bytes/(1024**2) )
		else:
			print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

		s.dev = cuda.Device(s.rank)
		s.ctx = s.dev.make_context()
		s.MAX_BLOCK = s.dev.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)


	def finalize(s):
		s.ctx.pop()


	def alloc_eh_fields(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.ex_gpu = cuda.to_device(f)
		s.ey_gpu = cuda.to_device(f)
		s.ez_gpu = cuda.to_device(f)
		s.hx_gpu = cuda.to_device(f)
		s.hy_gpu = cuda.to_device(f)
		s.hz_gpu = cuda.to_device(f)
		s.eh_fields = [s.ex_gpu, s.ey_gpu, s.ez_gpu, s.hx_gpu, s.hy_gpu, s.hz_gpu]


	def alloc_coeff_arrays(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.cex = np.ones_like(f)*0.5
		s.cex[:,-1,:] = 0
		s.cex[:,:,-1] = 0
		s.cey = np.ones_like(f)*0.5
		s.cey[:,:,-1] = 0
		s.cey[-1,:,:] = 0
		s.cez = np.ones_like(f)*0.5
		s.cez[-1,:,:] = 0
		s.cez[:,-1,:] = 0

		descr = cuda.ArrayDescriptor3D()
		descr.width = s.nz
		descr.height = s.ny
		descr.depth = s.nx
		descr.format = cuda.dtype_to_array_format(f.dtype)
		descr.num_channels = 1
		descr.flags = 0
		s.tcex_gpu = cuda.Array(descr)
		s.tcey_gpu = cuda.Array(descr)
		s.tcez_gpu = cuda.Array(descr)

		mcpy = cuda.Memcpy3D()
		mcpy.width_in_bytes = mcpy.src_pitch = f.strides[1]
		mcpy.src_height = mcpy.height = s.ny
		mcpy.depth = s.nx
		mcpy.set_src_host( s.cex )
		mcpy.set_dst_array( s.tcex_gpu )
		mcpy()
		mcpy.set_src_host( s.cey )
		mcpy.set_dst_array( s.tcey_gpu )
		mcpy()
		mcpy.set_src_host( s.cez )
		mcpy.set_dst_array( s.tcez_gpu )
		mcpy()


	def alloc_exchange_boundaries(s):
		s.ey_tmp = cuda.pagelocked_zeros((s.ny,s.nz),'f')
		s.ez_tmp = cuda.pagelocked_zeros_like(s.ey_tmp)
		s.hy_tmp = cuda.pagelocked_zeros_like(s.ey_tmp)
		s.hz_tmp = cuda.pagelocked_zeros_like(s.ey_tmp)


	def prepare_functions(s):
		# Constant variables (nx, ny, nz, nyz, Dx, Dy) are replaced by python string processing.
		kernels ="""
__global__ void update_h(int by0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y + by0;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty+1][tx+1] = ex[idx];
	sy[ty][tx+1] = ey[idx];
	sz[ty+1][tx] = ez[idx];
	if( tx == 0 && k > 0 ) {
		sx[ty+1][0] = ex[idx-1];
		sy[ty][0] = ey[idx-1];
	}
	if( ty == 0 && j > 0 ) {
		sx[0][tx+1] = ex[idx-nz];
		sz[0][tx] = ez[idx-nz];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[ty+1][tx+1] - sx[ty+1][tx] - sz[ty+1][tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[ty][tx+1] - ey[idx-nyz] - sx[ty+1][tx+1] + sx[ty][tx+1] );
}

texture<float, 3, cudaReadModeElementType> tcex;
texture<float, 3, cudaReadModeElementType> tcey;
texture<float, 3, cudaReadModeElementType> tcez;

__global__ void update_e(int by0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y + by0;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty][tx] = hx[idx];
	sy[ty][tx] = hy[idx];
	sz[ty][tx] = hz[idx];
	if( tx == Dx-1 && k < nz-1 ) {
		sx[ty][Dx] = hx[idx+1];
		sy[ty][Dx] = hy[idx+1];
	}
	if( ty == Dy-1 && j < ny-1 ) {
		sx[Dy][tx] = hx[idx+nz];
		sz[Dy][tx] = hz[idx+nz];
	}
	__syncthreads();

	ex[idx] += tex3D(tcex,k,j,i)*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i<nx-1 ) {
		ey[idx] += tex3D(tcey,k,j,i)*( sx[ty][tx+1] - sx[ty][tx] - hz[idx+nyz] + sz[ty][tx] );
		ez[idx] += tex3D(tcez,k,j,i)*( hy[idx+nyz] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
	}
}

__global__ void update_src(float tn, float *f) {
	int idx = threadIdx.x;
	//int ijk = (nx/2)*ny*nz + (ny/2)*nz + idx;
	int ijk = (nx-20)*ny*nz + (ny/2)*nz + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
		"""
		from pycuda.compiler import SourceModule
		mod = SourceModule( kernels.replace('Dx',str(s.Dx)).replace('Dy',str(s.Dy)).replace('nyz',str(s.ny*s.nz)).replace('nx',str(s.nx)).replace('ny',str(s.ny)).replace('nz',str(s.nz)) )
		s.updateH = mod.get_function("update_h")
		s.updateE = mod.get_function("update_e")
		s.updateE_src = mod.get_function("update_src")

		tcex = mod.get_texref("tcex")
		tcey = mod.get_texref("tcey")
		tcez = mod.get_texref("tcez")
		tcex.set_array(s.tcex_gpu)
		tcey.set_array(s.tcey_gpu)
		tcez.set_array(s.tcez_gpu)

		Bx, By = s.nz/s.Dx, s.nx*s.ny/s.Dy	# number of block
		s.MaxBy = s.MAX_BLOCK/Bx
		s.bpg_list = [(Bx,s.MaxBy) for i in range(By/s.MaxBy)]
		if By%s.MaxBy != 0: s.bpg_list.append( (Bx,By%s.MaxBy) )

		s.updateH.prepare("iPPPPPP", block=(s.Dx,s.Dy,1))
		s.updateE.prepare("iPPPPPP", block=(s.Dx,s.Dy,1), texrefs=[tcex,tcey,tcez])
		s.updateE_src.prepare("fP", block=(s.nz,1,1))


	def update_h(s):
		for i, bpg in enumerate(s.bpg_list): s.updateH.prepared_call(bpg, np.int32(i*s.MaxBy), *s.eh_fields)


	def update_e(s):
		for i, bpg in enumerate(s.bpg_list): s.updateE.prepared_call(bpg, np.int32(i*s.MaxBy), *s.eh_fields)


	def update_src(s, tn):
		s.updateE_src.prepared_call((1,1), np.float32(tn), s.ez_gpu)


	def mpi_exchange_boundary_h(s, mpi_direction):
		if 'f' in mpi_direction:
			comm.Recv(s.hy_tmp, s.rank-1, 0)
			comm.Recv(s.hz_tmp, s.rank-1, 1)
			cuda.memcpy_htod(int(s.hy_gpu), s.hy_tmp) 
			cuda.memcpy_htod(int(s.hz_gpu), s.hz_tmp) 
		if 'b' in mpi_direction:
			cuda.memcpy_dtoh(s.hy_tmp, int(s.hy_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32']) 
			cuda.memcpy_dtoh(s.hz_tmp, int(s.hz_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32']) 
			comm.Send(s.hy_tmp, s.rank+1, 0)
			comm.Send(s.hz_tmp, s.rank+1, 1)


	def mpi_exchange_boundary_e(s, mpi_direction):
		if 'f' in mpi_direction:
			cuda.memcpy_dtoh(s.ey_tmp, int(s.ey_gpu)) 
			cuda.memcpy_dtoh(s.ez_tmp, int(s.ez_gpu)) 
			comm.Send(s.ey_tmp, s.rank-1, 2)
			comm.Send(s.ez_tmp, s.rank-1, 3)
		if 'b' in mpi_direction:
			comm.Recv(s.ey_tmp, s.rank+1, 2)
			comm.Recv(s.ez_tmp, s.rank+1, 3)
			cuda.memcpy_htod(int(s.ey_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32'], s.ey_tmp) 
			cuda.memcpy_htod(int(s.ez_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32'], s.ez_tmp) 


nx, ny, nz = 512, 480, 480
tmax, tgap = 300, 10

fdtd = Fdtd3DGpu(nx, ny, nz)
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
	fdtd.mpi_exchange_boundary_h(mpi_direction)

	fdtd.update_e()
	fdtd.mpi_exchange_boundary_e(mpi_direction)

	if rank == 1: fdtd.update_src(tn)

	if tn%tgap == 0 and rank == 0:
		stop.record()
		stop.synchronize()
		flops[tn/tgap] = flop/stop.time_since(start)*1e-6
		print '[',datetime.now()-t1,']'," %d/%d (%d %%) %1.2f GFLOPS\r" % (tn, tmax, float(tn)/tmax*100, flops[tn/tgap]),
		sys.stdout.flush()
		start.record()

if rank == 0: print "\navg: %1.2f GFLOPS" % flops[2:-2].mean()
fdtd.finalize()

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

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3

