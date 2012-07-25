#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
__global__ void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x+1];
	float* sz = (float*) &sy[blockDim.x+1];

	sx[tx] = hx[idx];
	sy[tx] = hy[idx];
	sz[tx] = hz[idx];
	__syncthreads();

	float hy_x = sy[tx+1];
	float hz_x = sz[tx+1];
	if( tx == blockDim.x - 1 ) {
		hy_x = hy[idx+1];
		hz_x = hz[idx+1];
	}

	if( idx < nx*ny*(nz-1) ) {
		ex[idx] += cex[idx]*( hz[idx+nx] - sz[tx] - hy[idx+nx*ny] + sy[tx] );
		ey[idx] += cey[idx]*( hx[idx+nx*ny] - sx[tx] - hz_x + sz[tx] );
	}

	if( idx < nx*(ny*nz-1) )
		ez[idx] += cez[idx]*( hy_x - sy[tx] - hx[idx+nx] + sx[tx] );
}

__global__ void update_h(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *chx, float *chy, float *chz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x];
	float* sz = (float*) &sy[blockDim.x];

	sx[tx] = ex[idx];
	sy[tx] = ey[idx];
	sz[tx] = ez[idx];
	__syncthreads();

	float hy_x = sy[tx-1];
	float hz_x = sz[tx-1];
	if( tx == 0 ) {
		hy_x = ey[idx-1];
		hz_x = ez[idx-1];
	}

	if( idx > nx*ny ) {
		hx[idx] -= chx[idx]*( sz[tx] - ez[idx-nx] - sy[tx] + ey[idx-nx*ny] );
		hy[idx] -= chy[idx]*( sx[tx] - ex[idx-nx*ny] - sz[tx] + hz_x );
	}
	
	if( idx > nx ) 
		hz[idx] -= chz[idx]*( sy[tx] - hy_x - sx[tx] + ex[idx-nx] );
}

__global__ void update_src(int nx, int ny, int nz, int tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nz/2)*nx*ny + (ny/2)*nx + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
"""


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

	return cf


if __name__ == '__main__':
	nx, ny, nz = 256, 256, 240

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*12
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), 'f', order='F')

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device( set_c(f,(None,-1,-1)) )
	cey_gpu = cuda.to_device( set_c(f,(-1,None,-1)) )
	cez_gpu = cuda.to_device( set_c(f,(-1,-1,None)) )
	chx_gpu = cuda.to_device( set_c(f,(None,0,0)) )
	chy_gpu = cuda.to_device( set_c(f,(0,None,0)) )
	chz_gpu = cuda.to_device( set_c(f,(0,0,None)) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_e = mod.get_function("update_e")
	update_h = mod.get_function("update_h")
	update_src = mod.get_function("update_src")

	tpb = 256
	bpg = (nx*ny*nz)/tpb

	Db = (tpb,1,1)
	Dg = (bpg, 1)
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_e.prepare("iiiPPPPPPPPP", block=Db, shared=tpb*3*4)
	update_h.prepare("iiiPPPPPPPPP", block=Db, shared=tpb*3*4)
	update_src.prepare("iiiiP", block=(256,1,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.001)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	# main loop
	for tn in xrange(1, 1001):
		update_e.prepared_call(
				Dg, nnx, nny, nnz, 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

		update_h.prepared_call(
				Dg, nnx, nny, nnz, 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				chx_gpu, chy_gpu, chz_gpu)
			
		'''
		if tn%100 == 0:
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
