#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
__global__ void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Dx = blockDim.x;
	int Dy = blockDim.y;
	int sidx = ty*Dx + tx;
	int idx = (blockIdx.y*Dy + ty)*nz + blockIdx.x*Dx + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[Dx*Dy];
	float* sz = (float*) &sy[Dx*Dy];

	sx[sidx] = hx[idx];
	sy[sidx] = hy[idx];
	sz[sidx] = hz[idx];
	__syncthreads();

	float hx_z = sx[sidx+1];
	float hy_z = sy[sidx+1];
	float hx_y = sx[sidx+Dx];
	float hz_y = sz[sidx+Dx];
	if( tx == Dx-1 ) {
		hx_z = hx[idx+1];
		hy_z = hy[idx+1];
	}
	if( ty == Dy-1 ) {
		hx_y = hx[idx+nx];
		hz_y = hz[idx+nx];
	}

	if( idx < nx*ny*(nz-1) ) {
		ex[idx] += cex[idx]*( hz_y - sz[sidx] - hy_z + sy[sidx] );
		ey[idx] += cey[idx]*( hx_z - sx[sidx] - hz[idx+ny*nz] + sz[sidx] );
		ez[idx] += cez[idx]*( hy[idx+ny*nz] - sy[sidx] - hx_y + sx[sidx] );
	}
}

__global__ void update_h(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *chx, float *chy, float *chz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Dx = blockDim.x;
	int Dy = blockDim.y;
	int sidx = ty*Dx + tx;
	int idx = (blockIdx.y*Dy + ty)*nz + blockIdx.x*Dx + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[Dx*Dy];
	float* sz = (float*) &sy[Dx*Dy];

	sx[sidx] = ex[idx];
	sy[sidx] = ey[idx];
	sz[sidx] = ez[idx];
	__syncthreads();

	float ex_z = sx[sidx-1];
	float ey_z = sy[sidx-1];
	float ex_y = sx[sidx-Dx];
	float ez_y = sz[sidx-Dx];
	if( tx == 0 ) {
		ex_z = ex[idx-1];
		ey_z = ey[idx-1];
	}
	if( ty == 0 ) {
		ex_y = ex[idx-nx];
		ez_y = ez[idx-nz];
	}

	if( idx > ny*nz && idx < nx*ny*nz ) {
		hx[idx] -= chx[idx]*( sz[sidx] - ez_y - sy[sidx] + ey_z );
		hy[idx] -= chy[idx]*( sx[sidx] - ex_z - sz[sidx] + ez[idx-ny*nz] );
		hz[idx] -= chz[idx]*( sy[sidx] - ey[idx-ny*nz] - sx[sidx] + ex_y );
	}
}

__global__ void update_src(int nx, int ny, int nz, int tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nx/2)*ny*nz + (ny/2)*nz + idx;

	if( idx < nz ) f[ijk] += sin(0.1*tn);
}
"""


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

	return cf


if __name__ == '__main__':
	nx, ny, nz = 320, 320, 320

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	print 'mem %1.2f GB' % ( nx*ny*nz*4*12./(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz),'f')
	cf = np.zeros_like(f)

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device( set_c(cf,(None,-1,-1)) )
	cey_gpu = cuda.to_device( set_c(cf,(-1,None,-1)) )
	cez_gpu = cuda.to_device( set_c(cf,(-1,-1,None)) )
	chx_gpu = cuda.to_device( set_c(cf,(None,0,0)) )
	chy_gpu = cuda.to_device( set_c(cf,(0,None,0)) )
	chz_gpu = cuda.to_device( set_c(cf,(0,0,None)) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_e = mod.get_function("update_e")
	update_h = mod.get_function("update_h")
	update_src = mod.get_function("update_src")

	tpbx, tpby = 32, 16
	Db = (tpbx, tpby, 1)
	Dg = (nz/tpbx, nx*ny/tpby)
	smem = 3*tpbx*tpby*4
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_e.prepare("iiiPPPPPPPPP", block=Db, shared=smem)
	update_h.prepare("iiiPPPPPPPPP", block=Db, shared=smem)
	update_src.prepare("iiiiP", block=(512,1,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((nx,ny),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.001)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()
	
	# main loop
	for tn in xrange(1, 11):
		update_e.prepared_call(Dg, nnx, nny, nnz,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ez_gpu)

		update_h.prepared_call(Dg, nnx, nny, nnz, 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				chx_gpu, chy_gpu, chz_gpu)
		'''
		if tn%100 == 0:
			print 'tn =', tn
			cuda.memcpy_dtoh(f, ez_gpu)
			imsh.set_array( f[:,:,nz/2]**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
		'''
	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
