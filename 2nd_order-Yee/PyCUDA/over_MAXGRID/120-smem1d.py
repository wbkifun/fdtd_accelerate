#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
__global__ void update_e(int nx, int ny, int nz, int idx0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx + idx0;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x+1];
	float* sz = (float*) &sy[blockDim.x+1];

	sx[tx] = hx[idx];
	sy[tx] = hy[idx];
	sz[tx] = hz[idx];
	if( tx == blockDim.x - 1 ) {
		sx[tx+1] = hx[idx+1];
		sy[tx+1] = hy[idx+1];
	}
	__syncthreads();

	if( idx < nx*ny*(nz-1) ) {
		ex[idx] += cex[idx]*( hz[idx+nz] - sz[tx] - sy[tx+1] + sy[tx] );
		ey[idx] += cey[idx]*( sx[tx+1] - sx[tx] - hz[idx+ny*nz] + sz[tx] );
		ez[idx] += cez[idx]*( hy[idx+ny*nz] - sy[tx] - hx[idx+nz] + sx[tx] );
	}
}

__global__ void update_h(int nx, int ny, int nz, int idx0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *chx, float *chy, float *chz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx + idx0;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x+1];
	float* sz = (float*) &sy[blockDim.x+1];

	sx[tx+1] = ex[idx];
	sy[tx+1] = ey[idx];
	sz[tx] = ez[idx];
	if( tx == 0 ) {
		sx[0] = ex[idx-1];
		sy[0] = ey[idx-1];
	}
	__syncthreads();

	if( idx > ny*nz && idx < nx*ny*nz ) {
		hx[idx] -= chx[idx]*( sz[tx] - ez[idx-nz] - sy[tx+1] + sy[tx] );
		hy[idx] -= chy[idx]*( sx[tx+1] - sx[tx] - sz[tx] + ez[idx-ny*nz] );
		hz[idx] -= chz[idx]*( sy[tx+1] - ey[idx-ny*nz] - sx[tx+1] + ex[idx-nz] );
	}
}

__global__ void update_src(int nx, int ny, int nz, int tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nx/2)*ny*nz + (ny/2)*nz + idx;

	if( idx < nz ) f[ijk] += sin(0.1*tn);
}
"""


if __name__ == '__main__':
	nx, ny, nz = 512, 400, 400

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	print 'mem %1.2f GB' % ( nx*ny*nz*4*12./(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz),'f')
	cf = np.ones((nx,ny,nz),'f')*0.5

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
	update_e = mod.get_function("update_e")
	update_h = mod.get_function("update_h")
	update_src = mod.get_function("update_src")

	tpb = 512
	bpg = (nx*ny*nz)/tpb

	Db = (tpb,1,1)
	Dg_list = [ (bpg%MAX_GRID, 1) ]
	idx0_list = [ np.int32(0) ]
	ng = int( np.ceil( float(bpg)/MAX_GRID ) )
	for i in range(ng-1): 
		Dg_list.insert( 0, (MAX_GRID, 1) )
		idx0_list.append( np.int32(MAX_GRID*tpb*(i+1)) )

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_e.prepare("iiiiPPPPPPPPP", block=Db, shared=(3*tpb+2)*4)
	update_h.prepare("iiiiPPPPPPPPP", block=Db, shared=(3*tpb+2)*4)
	update_src.prepare("iiiiP", block=(512,1,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((nx,ny),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.001)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	'''
	start = cuda.Event()
	stop = cuda.Event()
	start.record()
	'''
	# main plot
	for tn in xrange(1, 101):
		for i in range(ng): update_e.prepared_call(Dg_list[i], 
				nnx, nny, nnz, idx0_list[i], 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ez_gpu)

		for i in range(ng): update_h.prepared_call(Dg_list[i], 
				nnx, nny, nnz, idx0_list[i], 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				chx_gpu, chy_gpu, chz_gpu)
		
	print 'tn =', tn
	cuda.memcpy_dtoh(f, ez_gpu)
	imsh.set_array( f[:,:,nz/2]**2 )
	show()
	#savefig('./png-wave/%.5d.png' % tstep) 
	'''
		
	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
	'''
