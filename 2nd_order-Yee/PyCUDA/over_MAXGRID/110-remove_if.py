#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
__global__ void update_e(int nx, int ny, int nz, int idx0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x + idx0;

	if( idx < nx*ny*(nz-1) ) {
		ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
		ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+ny*nz] + hz[idx] );
		ez[idx] += cez[idx]*( hy[idx+ny*nz] - hy[idx] - hx[idx+nz] + hx[idx] );
	}
}

__global__ void update_h(int nx, int ny, int nz, int idx0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *chx, float *chy, float *chz) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x + idx0;

	if( idx > ny*nz && idx < nx*ny*nz ) {
		hx[idx] -= chx[idx]*( ez[idx] - ez[idx-nz] - ey[idx] + ey[idx-1] );
		hy[idx] -= chy[idx]*( ex[idx] - ex[idx-1] - ez[idx] + ez[idx-ny*nz] );
		hz[idx] -= chz[idx]*( ey[idx] - ey[idx-ny*nz] - ex[idx] + ex[idx-nz] );
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
	nx, ny, nz = 512, 400, 400

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
	update_e.prepare("iiiiPPPPPPPPP", block=Db)
	update_h.prepare("iiiiPPPPPPPPP", block=Db)
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
	
	# main loop
	for tn in xrange(1, 301):
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
