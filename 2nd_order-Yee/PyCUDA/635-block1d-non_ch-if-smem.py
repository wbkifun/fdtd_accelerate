#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels =""" 
#define Dx 512

__global__ void update_h(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;
	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	__shared__ float s[3*Dx+2];
	float *sx = s;
	float *sy = &sx[Dx+1];
	float *sz = &sy[Dx+1];

	sx[tx] = ex[idx];
	sy[tx] = ey[idx];
	sz[tx] = ez[idx];
	if( tx == 0 ) {
		sy[tx-1] = ey[idx-1];
		sz[tx-1] = ez[idx-1];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-nx] - sy[tx] + ey[idx-nx*ny] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[tx] - ex[idx-nx*ny] - sz[tx] + sz[tx-1] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - sy[tx-1] - sx[tx] + ex[idx-nx] );
}

__global__ void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;
	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	__shared__ float s[3*Dx+2];
	float *sx = s;
	float *sy = &sx[Dx];
	float *sz = &sy[Dx+1];

	sx[tx] = hx[idx];
	sy[tx] = hy[idx];
	sz[tx] = hz[idx];
	if( tx == Dx-1 ) {
		sy[tx+1] = hy[idx+1];
		sz[tx+1] = hz[idx+1];
	}
	__syncthreads();

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nx] - sz[tx] - hy[idx+nx*ny] + sy[tx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nx*ny] - sx[tx] - sz[tx+1] + sz[tx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( sy[tx+1] - sy[tx] - hx[idx+nx] + sx[tx] );
}

__global__ void update_src(int nx, int ny, int nz, float tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nz/2)*nx*ny + (ny/2)*nx + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
"""


if __name__ == '__main__':
	nx, ny, nz = 300, 300, 304

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*9
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), dtype=np.float32, order='F')
	cf = np.ones_like(f)*0.5

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device(cf)
	cey_gpu = cuda.to_device(cf)
	cez_gpu = cuda.to_device(cf)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_h = mod.get_function("update_h")
	update_e = mod.get_function("update_e")
	update_src = mod.get_function("update_src")

	tpb = 512
	bpg = (nx*ny*nz)/tpb

	Db = (tpb,1,1)
	Dg = (bpg, 1)
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_h.prepare("iiiPPPPPP", block=Db, shared=(tpb*3+2)*4)
	update_e.prepare("iiiPPPPPPPPP", block=Db, shared=(tpb*3+2)*4)
	update_src.prepare("iiifP", block=(256,1,1))

	# prepare for plot
	'''
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),dtype=np.float32), cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
	colorbar()
	'''

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	# main loop
	for tn in xrange(1, 1000+1):
		update_h.prepared_call(
				Dg, nnx, nny, nnz,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		update_e.prepared_call(
				Dg, nnx, nny, nnz,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.float32(tn), ex_gpu)

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
