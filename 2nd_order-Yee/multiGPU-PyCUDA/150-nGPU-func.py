#!/usr/bin/env python
import numpy as np
import pycuda.driver as cuda
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cuda.init()
MAX_BLOCK = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
MAX_MEM = 4056	# MByte
ngpu = cuda.Device.count()
ctx = cuda.Device(rank).make_context()

def get_fdtd_functions_gpu(nx, ny, nz, Dx, Dy, tcex_gpu, tcey_gpu, tcez_gpu):
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
	mod = SourceModule( kernels.replace('Dx',str(Dx)).replace('Dy',str(Dy)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
	update_h = mod.get_function("update_h")
	update_e = mod.get_function("update_e")
	update_src = mod.get_function("update_src")

	tcex = mod.get_texref("tcex")
	tcey = mod.get_texref("tcey")
	tcez = mod.get_texref("tcez")
	tcex.set_array(tcex_gpu)
	tcey.set_array(tcey_gpu)
	tcez.set_array(tcez_gpu)

	update_h.prepare("iPPPPPP", block=(Dx,Dy,1))
	update_e.prepare("iPPPPPP", block=(Dx,Dy,1), texrefs=[tcex,tcey,tcez])
	update_src.prepare("fP", block=(nz,1,1))

	return update_h, update_e, update_src


def malloc_gpu_arrays(nx, ny, nz, cex, cey, cez):
	print 'rank= %d, (%d, %d, %d)' % (rank, nx, ny, nz),
	total_bytes = nx*ny*nz*4*9
	if total_bytes/(1024**3) == 0:
		print '%d MB' % ( total_bytes/(1024**2) )
	else:
		print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

	if nz%Dx != 0:
		print "Error: nz is not multiple of %d" % (Dx)
		sys.exit()
	if (nx*ny)%Dy != 0:
		print "Error: nx*ny is not multiple of %d" % (Dy)
		sys.exit()

	f = np.zeros((nx,ny,nz), 'f')
	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	descr = cuda.ArrayDescriptor3D()
	descr.width = nz
	descr.height = ny
	descr.depth = nx
	descr.format = cuda.dtype_to_array_format(f.dtype)
	descr.num_channels = 1
	descr.flags = 0
	tcex_gpu = cuda.Array(descr)
	tcey_gpu = cuda.Array(descr)
	tcez_gpu = cuda.Array(descr)

	mcopy = cuda.Memcpy3D()
	mcopy.width_in_bytes = mcopy.src_pitch = f.strides[1]
	mcopy.src_height = mcopy.height = ny
	mcopy.depth = nx

	mcopy.set_src_host( cex )
	mcopy.set_dst_array( tcex_gpu )
	mcopy()
	mcopy.set_src_host( cey )
	mcopy.set_dst_array( tcey_gpu )
	mcopy()
	mcopy.set_src_host( cez )
	mcopy.set_dst_array( tcez_gpu )
	mcopy()

	eh_fields = [ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu]
	tex_fields = [tcex_gpu, tcey_gpu, tcez_gpu]

	return eh_fields, tex_fields


def set_ce_arrays( *n_list ):
	f = np.ones((nx,ny,nz), 'f')*0.5
	cex = f.copy()
	cex[:,-1,:] = 0 
	cex[:,:,-1] = 0 
	cey = f.copy()
	cey[:,:,-1] = 0 
	cey[-1,:,:] = 0 
	cez = f.copy()
	cez[-1,:,:] = 0 
	cez[:,-1,:] = 0 

	return [cex, cey, cez]



n_list = [nx, ny, nz] = [512, 480, 480]
dim_list = [Dx, Dy] = [32, 16]
tmax, tgap = 100, 10

Bx, By = nz/Dx, nx*ny/Dy	# number of block
MBy = MAX_BLOCK/Bx
bpg_list = [(Bx,MBy) for i in range(By/MBy)]
if By%MBy != 0: bpg_list.append( (Bx,By%MBy) )

ce_fields = set_ce_arrays( *n_list )
eh_fields, tex_fields = malloc_gpu_arrays( *(n_list + ce_fields) )
update_h, update_e, update_src = get_fdtd_functions_gpu( *(n_list + dim_list + tex_fields) )

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

elif rank == 1:
	start, stop = cuda.Event(), cuda.Event()
	exec_time = {'update_h':np.zeros(tmax), 'mpi_recv_h':np.zeros(tmax), 'memcpy_htod_h':np.zeros(tmax), 'mpi_send_h':np.zeros(tmax), 'memcpy_dtoh_h':np.zeros(tmax), 
			'update_e':np.zeros(tmax), 'mpi_recv_e':np.zeros(tmax), 'memcpy_htod_e':np.zeros(tmax), 'mpi_send_e':np.zeros(tmax), 'memcpy_dtoh_e':np.zeros(tmax), 
			'src_e':np.zeros(tmax)}

# main loop
ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu = eh_fields
ey_tmp = np.zeros((ny,nz),'f')
ez_tmp = np.zeros_like(ey_tmp)
hy_tmp = np.zeros_like(ey_tmp)
hz_tmp = np.zeros_like(ey_tmp)
for tn in xrange(1, tmax+1):
	if rank == 1: start.record()
	for i, bpg in enumerate(bpg_list): update_h.prepared_call(bpg, np.int32(i*MBy), *eh_fields)

	if rank == 0:
		cuda.memcpy_dtoh(hy_tmp, int(hy_gpu)+(nx-1)*ny*nz*np.nbytes['float32']) 
		cuda.memcpy_dtoh(hz_tmp, int(hz_gpu)+(nx-1)*ny*nz*np.nbytes['float32']) 
		comm.Send(hy_tmp, 1, 20)
		comm.Send(hz_tmp, 1, 21)
	elif rank == 1:
		stop.record()
		stop.synchronize()
		exec_time['update_h'][tn-1] = stop.time_since(start)
		start.record()

		comm.Recv(hy_tmp, 0, 20)
		comm.Recv(hz_tmp, 0, 21)
		stop.record()
		stop.synchronize()
		exec_time['mpi_recv_h'][tn-1] = stop.time_since(start)
		start.record()

		cuda.memcpy_htod(int(hy_gpu), hy_tmp) 
		cuda.memcpy_htod(int(hz_gpu), hz_tmp) 
		stop.record()
		stop.synchronize()
		exec_time['memcpy_htod_h'][tn-1] = stop.time_since(start)
		start.record()

		cuda.memcpy_dtoh(hy_tmp, int(hy_gpu)+(nx-1)*ny*nz*np.nbytes['float32']) 
		cuda.memcpy_dtoh(hz_tmp, int(hz_gpu)+(nx-1)*ny*nz*np.nbytes['float32']) 
		stop.record()
		stop.synchronize()
		exec_time['memcpy_dtoh_h'][tn-1] = stop.time_since(start)
		start.record()

		comm.Send(hy_tmp, 2, 20)
		comm.Send(hz_tmp, 2, 21)
		stop.record()
		stop.synchronize()
		exec_time['mpi_send_h'][tn-1] = stop.time_since(start)
		start.record()
	elif rank == 2:
		comm.Recv(hy_tmp, 1, 20)
		comm.Recv(hz_tmp, 1, 21)
		cuda.memcpy_htod(int(hy_gpu), hy_tmp) 
		cuda.memcpy_htod(int(hz_gpu), hz_tmp) 

	for i, bpg in enumerate(bpg_list): update_e.prepared_call(bpg, np.int32(i*MBy), *eh_fields)

	if rank == 0:
		comm.Recv(ey_tmp, 1, 22)
		comm.Recv(ez_tmp, 1, 23)
		cuda.memcpy_htod(int(ey_gpu)+(nx-1)*ny*nz*np.nbytes['float32'], ey_tmp) 
		cuda.memcpy_htod(int(ez_gpu)+(nx-1)*ny*nz*np.nbytes['float32'], ez_tmp) 
	elif rank == 1:
		stop.record()
		stop.synchronize()
		exec_time['update_e'][tn-1] = stop.time_since(start)
		start.record()

		comm.Recv(ey_tmp, 2, 22)
		comm.Recv(ez_tmp, 2, 23)
		stop.record()
		stop.synchronize()
		exec_time['mpi_recv_e'][tn-1] = stop.time_since(start)
		start.record()

		cuda.memcpy_htod(int(ey_gpu)+(nx-1)*ny*nz*np.nbytes['float32'], ey_tmp) 
		cuda.memcpy_htod(int(ez_gpu)+(nx-1)*ny*nz*np.nbytes['float32'], ez_tmp) 
		stop.record()
		stop.synchronize()
		exec_time['memcpy_htod_e'][tn-1] = stop.time_since(start)
		start.record()

		cuda.memcpy_dtoh(ey_tmp, int(ey_gpu)) 
		cuda.memcpy_dtoh(ez_tmp, int(ez_gpu)) 
		stop.record()
		stop.synchronize()
		exec_time['memcpy_dtoh_e'][tn-1] = stop.time_since(start)
		start.record()

		comm.Send(ey_tmp, 0, 22)
		comm.Send(ez_tmp, 0, 23)
		stop.record()
		stop.synchronize()
		exec_time['mpi_send_e'][tn-1] = stop.time_since(start)
		start.record()
	elif rank == 2:
		cuda.memcpy_dtoh(ey_tmp, int(ey_gpu)) 
		cuda.memcpy_dtoh(ez_tmp, int(ez_gpu)) 
		comm.Send(ey_tmp, 1, 22)
		comm.Send(ez_tmp, 1, 23)

	if rank == 1: 
		update_src.prepared_call((1,1), np.float32(tn), ez_gpu)
		stop.record()
		stop.synchronize()
		exec_time['src_e'][tn-1] = stop.time_since(start)


	if tn%tgap == 0 and rank == 0:
		stop.record()
		stop.synchronize()
		flops[tn/tgap] = flop/stop.time_since(start)*1e-6
		print '[',datetime.now()-t1,']'," %d/%d (%d %%) %1.2f GFLOPS\r" % (tn, tmax, float(tn)/tmax*100, flops[tn/tgap]),
		sys.stdout.flush()
		start.record()

if rank == 0: 
	print "\navg: %1.2f GFLOPS" % flops[2:-2].mean()

if rank == 1:
	total = np.zeros(tmax)
	for key in exec_time.iterkeys(): total[:] += exec_time[key][:]
	for key in exec_time.iterkeys():
		print key, ':\t %1.2f %%' % ( exec_time[key][2:-2].sum()/total[2:-2].sum()*100 )

	print "%1.2f GFLOPS\r" % ( (tmax-4)*3*nx*ny*nz*30/total[2:-2].sum()*1e-6 )

g = cuda.pagelocked_zeros((nx,ny,nz),'f')
cuda.memcpy_dtoh(g, ez_gpu)
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

ctx.pop()
