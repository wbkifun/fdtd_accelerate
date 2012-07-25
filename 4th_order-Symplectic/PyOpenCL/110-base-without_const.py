#!/usr/bin/env python

import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
MAX_GRID = 65535

kernels =""" 
__kernel void update_h(int nx, int ny, int nz, float cl, 
		__global float *ex, __global float *ey, __global float *ez, 
		__global float *hx, __global float *hy, __global float *hz, 
		__global float *chx, __global float *chy, __global float *chz) {
	int idx = get_global_id(0);
	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	if( j>1 && j<ny-1 && k>1 && k<nz-1 ) 
		hx[idx] -= cl*chx[idx]*( 27*( ez[idx] - ez[idx-nx] - ey[idx] + ey[idx-nx*ny] )
								- ( ez[idx+nx] - ez[idx-2*nx] - ey[idx+nx*ny] + ey[idx-2*nx*ny] ) );
	if( i>1 && i<nx-1 && k>1 && k<nz-1 ) 
		hy[idx] -= cl*chy[idx]*( 27*( ex[idx] - ex[idx-nx*ny] - ez[idx] + ez[idx-1] )
								- ( ex[idx+nx*ny] - ex[idx-2*nx*ny] - ez[idx+1] + ez[idx-2] ) );
	if( i>1 && i<nx-1 && j>1 && j<ny-1 ) 
		hz[idx] -= cl*chz[idx]*( 27*( ey[idx] - ey[idx-1] - ex[idx] + ex[idx-nx] )
								- ( ey[idx+1] - ey[idx-2] - ex[idx+nx] + ex[idx-2*nx] ) );
}

__kernel void update_e(int nx, int ny, int nz, float dl, 
		__global float *ex, __global float *ey, __global float *ez, 
		__global float *hx, __global float *hy, __global float *hz, 
		__global float *cex, __global float *cey, __global float *cez) {
	int idx = get_global_id(0);
	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	if( j>0 && j<ny-2 && k>0 && k<nz-2 ) 
		ex[idx] += dl*cex[idx]*( 27*( hz[idx+nx] - hz[idx] - hy[idx+nx*ny] + hy[idx] )
								- ( hz[idx+2*nx] - hz[idx-nx] - hy[idx+2*nx*ny] + hy[idx-nx*ny] ) );
	if( i>0 && i<nx-2 && k>0 && k<nz-2 ) 
		ey[idx] += dl*cey[idx]*( 27*( hx[idx+nx*ny] - hx[idx] - hz[idx+1] + hz[idx] )
								- ( hx[idx+2*nx*ny] - hx[idx-nx*ny] - hz[idx+2] + hz[idx-1] ) );
	if( i>0 && i<nx-2 && j>0 && j<ny-2 ) 
		ez[idx] += dl*cez[idx]*( 27*( hy[idx+1] - hy[idx] - hx[idx+nx] + hx[idx] )
								- ( hy[idx+2] - hy[idx-1] - hx[idx+2*nx] + hx[idx-nx] ) );
}

__kernel void update_src(int nx, int ny, int nz, float tn, 
		__global float *f) {
	int idx = get_global_id(0);
	int ijk = (nz/2)*nx*ny + (ny/2)*nx + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
"""


if __name__ == '__main__':
	nx, ny, nz = 256, 256, 240
	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)

	S = 0.743	# Courant stability factor
	# symplectic integrator coefficients
	c1, c2, c3 = 0.17399689146541, -0.12038504121430, 0.89277629949778
	d1, d2 = 0.62337932451322, -0.12337932451322
	clst = np.array([c1,c2,c3,c2,c1], dtype=np.float32)
	dlst = np.array([d1,d2,d2,d1,0], dtype=np.float32)

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*12
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), 'f', order='F')
	#f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz),order='F')
	cf = np.ones_like(f)*(S/24)

	mf = cl.mem_flags
	ex_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	ey_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	ez_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	hx_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	hy_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	hz_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)

	cex_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	cey_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	cez_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	chx_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	chy_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	chz_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)

	# prepare kernels
	prg = cl.Program(ctx, kernels).build()

	Gs = (nx*ny*nz,)	# global size
	Ls = (512,)			# local size

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.01)
	colorbar()

	# measure kernel execution time
	from datetime import datetime
	t1 = datetime.now()

	# main loop
	for tn in xrange(1, 100+1):
		for sn in xrange(4):
			prg.update_h(
					queue, Gs, nnx, nny, nnz, clst[sn],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
					chx_gpu, chy_gpu, chz_gpu, local_size=Ls)

			prg.update_e(
					queue, Gs, nnx, nny, nnz, dlst[sn],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
					cex_gpu, cey_gpu, cez_gpu, local_size=Ls)

			prg.update_src(queue, (nx,), nnx, nny, nnz, np.float32(tn+sn/5.), ex_gpu)

		prg.update_h(
				queue, Gs, nnx, nny, nnz, clst[4],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				chx_gpu, chy_gpu, chz_gpu, local_size=Ls)

		if tn%10 == 0:
		#if tn == 100:
			print 'tn =', tn
			cl.enqueue_read_buffer(queue, ex_gpu, f)
			imsh.set_array( f[nx/2,:,:]**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 

	cl.enqueue_read_buffer(queue, ex_gpu, f).wait()
	print datetime.now() - t1
