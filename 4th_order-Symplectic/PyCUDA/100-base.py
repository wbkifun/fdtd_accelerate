#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels =""" 
__global__ void update_h(int nx, int ny, int nz, float cl, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *chx, float *chy, float *chz) {
	int idx, i, j, k;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	k = idx/(nx*ny);
	j = (idx - k*nx*ny)/nx;
	i = idx%nx;

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

__global__ void update_e(int nx, int ny, int nz, float dl, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx, i, j, k;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	k = idx/(nx*ny);
	j = (idx - k*nx*ny)/nx;
	i = idx%nx;

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
	#f = np.zeros((nx,ny,nz), 'f', order='F')
	f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz),order='F')
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

	tpb = 256
	bpg = (nx*ny*nz)/tpb

	Db = (tpb,1,1)
	Dg = (bpg, 1)
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_e.prepare("iiifPPPPPPPPP", block=Db)
	update_h.prepare("iiifPPPPPPPPP", block=Db)
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
