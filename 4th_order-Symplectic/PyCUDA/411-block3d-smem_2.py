#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels ="""
#define Dx 16
#define Dy 8
#define Dz 4

__global__ void update_h(int nx, int ny, int nz, float cl, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *chx, float *chy, float *chz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int gDy = ny/Dy;
	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int i = blockIdx.x*Dx+tx;
	int j = by*Dy+ty;
	int k = bz*Dz+tz;
	int idx = i + j*nx + k*nx*ny;

	__shared__ float sx[Dx][Dy+3][Dz+3];
	__shared__ float sy[Dx+3][Dy][Dz+3];
	__shared__ float sz[Dx+3][Dy+3][Dz];

	sx[tx][ty+2][tz+2] = ex[idx];
	sy[tx+2][ty][tz+2] = ey[idx];
	sz[tx+2][ty+2][tz] = ez[idx];

	if (tx == 0 && i > 1) {
		sy[0][ty][tz+2] = ey[idx-2];
		sz[0][ty+2][tz] = ez[idx-2];
	}
	if (tx == 0 && i > 0) {
		sy[1][ty][tz+2] = ey[idx-1];
		sz[1][ty+2][tz] = ez[idx-1];
	}
	if (ty == 0 && j > 1) {
		sx[tx][0][tz+2] = ex[idx-2*nx];
		sz[tx+2][0][tz] = ez[idx-2*nx];
	}
	if (ty == 0 && j > 0) {
		sx[tx][1][tz+2] = ex[idx-nx];
		sz[tx+2][1][tz] = ez[idx-nx];
	}
	if (tz == 0 && k > 1) {
		sx[tx][ty+2][0] = ex[idx-2*nx*ny];
		sy[tx+2][ty][0] = ey[idx-2*nx*ny];
	}
	if (tz == 0 && k > 0) {
		sx[tx][ty+2][1] = ex[idx-nx*ny];
		sy[tx+2][ty][1] = ey[idx-nx*ny];
	}
	if (tx == Dx-1 && i < nx-1) {
		sy[tx+3][ty][tz+2] = ey[idx+1];
		sz[tx+3][ty+2][tz] = ez[idx+1];
	}
	if (ty == Dy-1 && j < ny-1) {
		sx[tx][ty+3][tz+2] = ex[idx+nx];
		sz[tx+2][ty+3][tz] = ez[idx+nx];
	}
	if (tz == Dz-1 && k < nz-1) {
		sx[tx][ty+2][tz+3] = ex[idx+nx*ny];
		sy[tx+2][ty][tz+3] = ey[idx+nx*ny];
	}
	__syncthreads();

	if( j>1 && j<ny-1 && k>1 && k<nz-1 ) 
		hx[idx] -= cl*chx[idx]*( 27*( ez[idx] - ez[idx-nx] - ey[idx] + ey[idx-nx*ny] )
								- ( ez[idx+nx] - ez[idx-2*nx] - ey[idx+nx*ny] + ey[idx-2*nx*ny] ) );
	if( i>1 && i<nx-1 && k>1 && k<nz-1 ) 
		hy[idx] -= cl*chy[idx]*( 27*( sx[tx][ty+2][tz+2] - sx[tx][ty+2][tz+1] - sz[tx+2][ty+2][tz] + sz[tx+1][ty+2][tz] )
								- ( sx[tx][ty+2][tz+3] - sx[tx][ty+2][tz] - sz[tx+3][ty+2][tz] + sz[tx][ty+2][tz] ) );
	if( i>1 && i<nx-1 && j>1 && j<ny-1 ) 
		hz[idx] -= cl*chz[idx]*( 27*( sy[tx+2][ty][tz+2] - sy[tx+1][ty][tz+2] - sx[tx][ty+2][tz+2] + sx[tx][ty+1][tz+2] )
								- ( sy[tx+3][ty][tz+2] - sy[tx][ty][tz+2] - sx[tx][ty+3][tz+2] + sx[tx][ty][tz+2] ) );
}

__global__ void update_e(int nx, int ny, int nz, float dl, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int gDy = ny/Dy;
	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int i = blockIdx.x*Dx+tx;
	int j = by*Dy+ty;
	int k = bz*Dz+tz;
	int idx = i + j*nx + k*nx*ny;

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

__global__ void update_src(int nx, int ny, int nz, float tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nz/2)*nx*ny + (ny/2)*nx + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
"""


if __name__ == '__main__':
	nx, ny, nz = 256, 256, 240

	S = 0.743	# Courant stability factor
	# symplectic integrator coefficients
	c1, c2, c3 = 0.17399689146541, -0.12038504121430, 0.89277629949778
	d1, d2 = 0.62337932451322, -0.12337932451322
	cl = np.array([c1,c2,c3,c2,c1], dtype=np.float32)
	dl = np.array([d1,d2,d2,d1,0], dtype=np.float32)

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

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device(cf)
	cey_gpu = cuda.to_device(cf)
	cez_gpu = cuda.to_device(cf)
	chx_gpu = cuda.to_device(cf)
	chy_gpu = cuda.to_device(cf)
	chz_gpu = cuda.to_device(cf)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_h = mod.get_function("update_h")
	update_e = mod.get_function("update_e")
	update_src = mod.get_function("update_src")

	Db = (16,8,4)
	Dg = (nx/16, ny*nz/(8*4))
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_h.prepare("iiifPPPPPPPPP", block=Db)
	update_e.prepare("iiifPPPPPPPPP", block=Db)
	update_src.prepare("iiifP", block=(256,1,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.01)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	# main loop
	for tn in xrange(1, 100+1):
		for sn in xrange(4):
			update_h.prepared_call(
					Dg, nnx, nny, nnz, cl[sn],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
					chx_gpu, chy_gpu, chz_gpu)

			update_e.prepared_call(
					Dg, nnx, nny, nnz, dl[sn],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
					cex_gpu, cey_gpu, cez_gpu)

			update_src.prepared_call((1,1), nnx, nny, nnz, np.float32(tn+sn/5.), ex_gpu)

		update_h.prepared_call(
				Dg, nnx, nny, nnz, cl[4],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				chx_gpu, chy_gpu, chz_gpu)

		'''	
		if tn%10 == 0:
		#if tn == 100:
			print 'tn =', tn
			cuda.memcpy_dtoh(f, ex_gpu)
			imsh.set_array( f[nx/2,:,:]**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
		'''	

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
