#!/usr/bin/env python

# Constant variables (nx, ny, nz, nxy, TPB) are replaced by python string processing.
kernels = """
#define Dx 16
#define Dy 16

__global__ void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int k = idx/nxy;
	int j = (idx - k*nxy)/nx;
	int i = idx%nx;

	__shared__ float s[3*TPB+2];
	float *sx = s;
	float *sy = &sx[TPB+1];
	float *sz = &sy[TPB+1];

	sx[tx] = ex[idx];
	sy[tx] = ey[idx];
	sz[tx] = ez[idx];
	if( tx == 0 ) {
		sy[-1] = ey[idx-1];
		sz[-1] = ez[idx-1];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-nx] - sy[tx] + ey[idx-nxy] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[tx] - ex[idx-nxy] - sz[tx] + sz[tx-1] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - sy[tx-1] - sx[tx] + ex[idx-nx] );
}

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int by = blockIdx.y;

	int k = (by*Dy)/ny;
	int j = (by*Dy + ty)%ny;

	__shared__ float sx[Dy+1][Dx];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx+1];

	int i,idx;
	for( int bx=0; bx<nx/Dx; bx++ ) {
		i = bx*Dx + tx;
		idx = (by*Dy + ty)*nx + bx*Dx + tx;

		sx[ty][tx] = hx[idx];
		sy[ty][tx] = hy[idx];
		sz[ty][tx] = hz[idx];
		if( tx==Dx-1 && i<nx-1 ) {
			sy[ty][Dx] = hy[idx+1];
			sz[ty][Dx] = hz[idx+1];
		}
		if( ty==Dy-1 && j<ny-1 ) {
			sx[Dx][tx] = hx[idx+nx];
			sz[Dx][tx] = hz[idx+nx];
		}
		__syncthreads();

		if( k<nz-1 ) ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - hy[idx+nxy] + sy[ty][tx] );
		if( k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - sx[ty][tx] - sz[ty][tx+1] + sz[ty][tx] );
		ez[idx] += cez[idx]*( sy[ty][tx+1] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
		__syncthreads();
	}
}

__global__ void update_src(float tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nz/2)*nxy + (ny/2)*nx + idx;

	f[ijk] += sin(0.1*tn);
}
"""

import numpy as np
import sys
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

def set_c(cf, ax, initval=0.5):
	cf[:,:,:] = initval
	if 'x' in ax: cf[-1,:,:] = 0
	if 'y' in ax: cf[:,-1,:] = 0
	if 'z' in ax: cf[:,:,-1] = 0

	return cf


if __name__ == '__main__':
	nx, ny, nz = 240, 256, 256
	tpb = 256
	bpg = (nx*ny*nz)/tpb

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*9
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

	cex_gpu = cuda.to_device( set_c(f,'yz') )
	cey_gpu = cuda.to_device( set_c(f,'zx') )
	cez_gpu = cuda.to_device( set_c(f,'xy') )

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule( kernels.replace('TPB',str(tpb)).replace('nxy',str(nx*ny)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
	update_h = mod.get_function("update_h")
	update_e = mod.get_function("update_e")
	update_src = mod.get_function("update_src")

	update_h.prepare("PPPPPP", block=(tpb,1,1))
	update_e.prepare("PPPPPPPPP", block=(16,16,1))
	update_src.prepare("fP", block=(nx,1,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	tmax = 300
	# main loop
	for tn in xrange(1, tmax+1):
		update_h.prepared_call(
				(bpg,1), ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		update_e.prepared_call(
				(1,ny*nz/16), ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), np.float32(tn), ex_gpu)

		
		if tn%10 == 0:
		#if tn == 100:
			print "tstep =\t%d/%d (%d %%)\r" % (tn, tmax, float(tn)/tmax*100),
			sys.stdout.flush()
			cuda.memcpy_dtoh(f, ex_gpu)
			imsh.set_array( f[nx/2,:,:].T**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
