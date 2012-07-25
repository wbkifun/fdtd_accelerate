#!/usr/bin/env python

kernels ="""
__global__ void update_h(int nx, int ny, int nz, int nyz, int idx0, float *hx, float *hy, float *hz, float *ex, float *ey, float *ez) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx + idx0;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j>0 && k>0 ) hx[idx] -= 0.5*( ez[idx] - ez[idx-nz] - ey[idx] + ey[idx-1] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( ex[idx] - ex[idx-1] - ez[idx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( ey[idx] - ey[idx-nyz] - ex[idx] + ex[idx-nz] );
}

__global__ void update_e(int nx, int ny, int nz, int nyz, int idx0, float *hx, float *hy, float *hz, float *ex, float *ey, float *ez, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx + idx0;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

__global__ void update_src(int nx, int ny, int nz, int nyz, float tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nx/2)*nyz + (ny/2)*nz + idx;

	if( idx < nz ) f[ijk] += sin(0.1*tn);
}
  
__global__ void init_zero(int n, int idx0, float *f) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx + idx0;
	if( idx < n ) f[idx] = 0;
}
"""
  
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import sys

MAX_GRID = 65535


if __name__ == '__main__':
	nx, ny, nz = 320, 480, 480
	n = nx*ny*nz
	tmax = 300

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*9
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), 'f')
	cf = np.ones_like(f)*0.5

	hx_gpu = cuda.mem_alloc(f.nbytes)
	hy_gpu = cuda.mem_alloc(f.nbytes)
	hz_gpu = cuda.mem_alloc(f.nbytes)
	ex_gpu = cuda.mem_alloc(f.nbytes)
	ey_gpu = cuda.mem_alloc(f.nbytes)
	ez_gpu = cuda.mem_alloc(f.nbytes)

	cex_gpu = cuda.to_device( cf )
	cey_gpu = cuda.to_device( cf )
	cez_gpu = cuda.to_device( cf )

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule( kernels )
	update_h = mod.get_function("update_h")
	update_e = mod.get_function("update_e")
	update_src = mod.get_function("update_src")
	init_zero = mod.get_function("init_zero")

	ng = 6						# number of grid
	tpb = 256					# threads per block
	bpg = (nx*ny*nz)/tpb/ng;	# blocks per grid

	update_h.prepare("iiiiiPPPPPP", block=(tpb,1,1))
	update_e.prepare("iiiiiPPPPPPPPP", block=(tpb,1,1))
	update_src.prepare("iiiifP", block=(nz,1,1))

	# initialize gpu arrays
	for i in xrange(ng):
		init_zero(np.int32(nx*ny*nz), np.int32(i*bpg*tpb), hx_gpu, block=(tpb,1,1), grid=(bpg,1))
		init_zero(np.int32(nx*ny*nz), np.int32(i*bpg*tpb), hy_gpu, block=(tpb,1,1), grid=(bpg,1))
		init_zero(np.int32(nx*ny*nz), np.int32(i*bpg*tpb), hz_gpu, block=(tpb,1,1), grid=(bpg,1))
		init_zero(np.int32(nx*ny*nz), np.int32(i*bpg*tpb), ex_gpu, block=(tpb,1,1), grid=(bpg,1))
		init_zero(np.int32(nx*ny*nz), np.int32(i*bpg*tpb), ey_gpu, block=(tpb,1,1), grid=(bpg,1))
		init_zero(np.int32(nx*ny*nz), np.int32(i*bpg*tpb), ez_gpu, block=(tpb,1,1), grid=(bpg,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((nx,ny),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	# main loop
	for tn in xrange(1, tmax+1):
		for i in range(ng):
			update_h.prepared_call( (bpg,1), np.int32(nx), np.int32(ny), np.int32(nz), np.int32(ny*nz), np.int32(i*bpg*tpb), hx_gpu, hy_gpu, hz_gpu, ex_gpu, ey_gpu, ez_gpu)

		for i in range(ng):
			update_e.prepared_call( (bpg,1), np.int32(nx), np.int32(ny), np.int32(nz), np.int32(ny*nz), np.int32(i*bpg*tpb), hx_gpu, hy_gpu, hz_gpu, ex_gpu, ey_gpu, ez_gpu, cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), np.int32(nx), np.int32(ny), np.int32(nz), np.int32(ny*nz), np.float32(tn), ez_gpu)

		if tn%10 == 0:
			print "tn =\t%d/%d (%d %%)\r" % (tn, tmax, float(tn)/tmax*100),
			sys.stdout.flush()
			cuda.memcpy_dtoh(f, ez_gpu)
			imsh.set_array( f[:,:,nz/2].T**2 )
			draw()
			#savefig('./png/%.5d.png' % tn) 

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
