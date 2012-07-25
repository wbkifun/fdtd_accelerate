#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
__global__ void update_e(int nx, int ny, int nz, int gDy, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int idx = (bz*blockDim.z + threadIdx.z)*nx*ny + (by*blockDim.y + threadIdx.y)*nx + (blockIdx.x*blockDim.x + threadIdx.x);

	if( idx < nx*ny*(nz-1) ) {
		ex[idx] += cex[idx]*( hz[idx+nx] - hz[idx] - hy[idx+nx*ny] + hy[idx] );
		ey[idx] += cey[idx]*( hx[idx+nx*ny] - hx[idx] - hz[idx+1] + hz[idx] );
	}

	if( idx < nx*(ny*nz-1) )
		ez[idx] += cez[idx]*( hy[idx+1] - hy[idx] - hx[idx+nx] + hx[idx] );
}

__global__ void update_h(int nx, int ny, int nz, int gDy, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *chx, float *chy, float *chz) {
	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int idx = (bz*blockDim.z + threadIdx.z)*nx*ny + (by*blockDim.y + threadIdx.y)*nx + (blockIdx.x*blockDim.x + threadIdx.x);

	if( idx > nx*ny ) {
		hx[idx] -= chx[idx]*( ez[idx] - ez[idx-nx] - ey[idx] + ey[idx-nx*ny] );
		hy[idx] -= chy[idx]*( ex[idx] - ex[idx-nx*ny] - ez[idx] + ez[idx-1] );
	}
	
	if( idx > nx ) 
		hz[idx] -= chz[idx]*( ey[idx] - ey[idx-1] - ex[idx] + ex[idx-nx] );
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


	Db = (16,4,4)
	Dg = (nx/16, ny*nz/(4*4))
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	gDy = np.int32(ny/Db[1])
	update_e.prepare("iiiiPPPPPPPPP", block=Db)
	update_h.prepare("iiiiPPPPPPPPP", block=Db)
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
				Dg, nnx, nny, nnz, gDy,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

		update_h.prepared_call(
				Dg, nnx, nny, nnz, gDy,
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
