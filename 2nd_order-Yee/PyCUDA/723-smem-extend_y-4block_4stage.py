#!/usr/bin/env python

# Constant variables (nx, ny, nz, nyz, TPB) are replaced by python string processing.
kernels ="""
__global__ void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	__shared__ float s[3*TPB+2];
	float *sz = s;
	float *sy = &sz[TPB+1];
	float *sx = &sy[TPB+1];

	sz[tx] = ez[idx];
	sy[tx] = ey[idx];
	sx[tx] = ex[idx];
	if( tx == 0 ) {
		sy[tx-1] = ey[idx-1];
		sx[tx-1] = ex[idx-1];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-nz] - sy[tx] + sy[tx-1] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[tx] - sx[tx-1] - sz[tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - ey[idx-nyz] - sx[tx] + ex[idx-nz] );
}

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = 4*TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	__shared__ float s[4*TPB];
	float tmp_ezx, tmp_ey;

	//------------------------------------------------------------------------------
	// first thread block
	//------------------------------------------------------------------------------
	s[tx] = hx[idx];
	s[tx+TPB] = hx[idx+TPB];
	__syncthreads();
	tmp_ezx = s[tx+TPB] - s[tx];
	tmp_ey = s[tx+1] - s[tx];

	s[tx+2*TPB] = hy[idx];
	s[tx+3*TPB] = hy[idx+TPB];
	__syncthreads();
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - s[tx+2*TPB] - tmp_ezx );
	tmp_ezx = s[tx+2*TPB+1] - s[tx+2*TPB];

	s[tx] = hz[idx];
	__syncthreads();
	s[tx+2*TPB] = hz[idx+TPB];
	__syncthreads();
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( tmp_ey - hz[idx+nyz] + s[tx] );
	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( s[tx+2*TPB] - s[tx] - tmp_ezx );

	//------------------------------------------------------------------------------
	// 2nd thread block
	//------------------------------------------------------------------------------
	__syncthreads();
	s[tx] = s[tx+TPB];
	__syncthreads();
	s[tx+TPB] = hx[idx+2*TPB];
	__syncthreads();
	tmp_ezx = s[tx+TPB] - s[tx];
	tmp_ey = s[tx+1] - s[tx];

	__syncthreads();
	s[tx] = s[tx+2*TPB];
	__syncthreads();
	s[tx+2*TPB] = s[tx+3*TPB];
	__syncthreads();
	s[tx+3*TPB] = hy[idx+2*TPB];
	__syncthreads();
	if( i<nx-1 && j<ny-1 ) ez[idx+TPB] += cez[idx+TPB]*( hy[idx+nyz+TPB] - s[tx+2*TPB] - tmp_ezx );
	tmp_ezx = s[tx+2*TPB+1] - s[tx+2*TPB];

	__syncthreads();
	s[tx+2*TPB] = hz[idx+2*TPB];
	if( i<nx-1 && k<nz-1 ) ey[idx+TPB] += cey[idx+TPB]*( tmp_ey - hz[idx+nyz+TPB] + s[tx] );
	__syncthreads();
	if( j<ny-1 && k<nz-1 ) ex[idx+TPB] += cex[idx+TPB]*( s[tx+2*TPB] - s[tx] - tmp_ezx );

	//------------------------------------------------------------------------------
	// 3rd thread block
	//------------------------------------------------------------------------------
	__syncthreads();
	s[tx] = s[tx+TPB];
	__syncthreads();
	s[tx+TPB] = hx[idx+3*TPB];
	__syncthreads();
	tmp_ezx = s[tx+TPB] - s[tx];
	tmp_ey = s[tx+1] - s[tx];

	__syncthreads();
	s[tx] = s[tx+2*TPB];
	__syncthreads();
	s[tx+2*TPB] = s[tx+3*TPB];
	__syncthreads();
	s[tx+3*TPB] = hy[idx+3*TPB];
	__syncthreads();
	if( i<nx-1 && j<ny-1 ) ez[idx+2*TPB] += cez[idx+2*TPB]*( hy[idx+nyz+2*TPB] - s[tx+2*TPB] - tmp_ezx );
	tmp_ezx = s[tx+2*TPB+1] - s[tx+2*TPB];

	__syncthreads();
	s[tx+2*TPB] = hz[idx+3*TPB];
	if( i<nx-1 && k<nz-1 ) ey[idx+2*TPB] += cey[idx+2*TPB]*( tmp_ey - hz[idx+nyz+2*TPB] + s[tx] );
	__syncthreads();
	if( j<ny-1 && k<nz-1 ) ex[idx+2*TPB] += cex[idx+2*TPB]*( s[tx+2*TPB] - s[tx] - tmp_ezx );

	//------------------------------------------------------------------------------
	// 4th thread block
	//------------------------------------------------------------------------------
	__syncthreads();
	s[tx] = s[tx+TPB];
	__syncthreads();
	s[tx+TPB] = hx[idx+4*TPB];
	__syncthreads();
	tmp_ezx = s[tx+TPB] - s[tx];
	tmp_ey = s[tx+1] - s[tx];

	__syncthreads();
	s[tx] = s[tx+2*TPB];
	__syncthreads();
	s[tx+2*TPB] = s[tx+3*TPB];
	__syncthreads();
	s[tx+3*TPB] = hy[idx+4*TPB];
	__syncthreads();
	if( i<nx-1 && j<ny-1 ) ez[idx+3*TPB] += cez[idx+3*TPB]*( hy[idx+nyz+3*TPB] - s[tx+2*TPB] - tmp_ezx );
	tmp_ezx = s[tx+2*TPB+1] - s[tx+2*TPB];

	__syncthreads();
	s[tx+2*TPB] = hz[idx+4*TPB];
	if( i<nx-1 && k<nz-1 ) ey[idx+3*TPB] += cey[idx+3*TPB]*( tmp_ey - hz[idx+nyz+3*TPB] + s[tx] );
	__syncthreads();
	if( j<ny-1 && k<nz-1 ) ex[idx+3*TPB] += cex[idx+3*TPB]*( s[tx+2*TPB] - s[tx] - tmp_ezx );
}

__global__ void update_src(float tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nx/2)*ny*nz + (ny/2)*nz + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
"""
  
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

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
	f = np.zeros((nx,ny,nz), 'f')
	cf = np.ones_like(f)*0.5

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
	mod = SourceModule( kernels.replace('TPB',str(tpb)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
	update_h = mod.get_function("update_h")
	update_e = mod.get_function("update_e")
	update_src = mod.get_function("update_src")

	update_h.prepare("PPPPPP", block=(tpb,1,1))
	update_e.prepare("PPPPPPPPP", block=(tpb,1,1))
	update_src.prepare("fP", block=(nz,1,1))

	'''
	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((nx,ny),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
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
				(bpg,1), ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		update_e.prepared_call(
				(bpg/4,1), ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu)

		update_src.prepared_call((1,1), np.float32(tn), ez_gpu)

		'''
		if tn%10 == 0:
		#if tn == 100:
			print 'tn =', tn
			cuda.memcpy_dtoh(f, ez_gpu)
			imsh.set_array( f[:,:,nz/2].T**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
		'''

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
