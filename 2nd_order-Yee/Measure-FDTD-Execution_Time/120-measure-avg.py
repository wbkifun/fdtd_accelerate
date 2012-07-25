#!/usr/bin/env python

# Constant variables (nx, ny, nz, nyz, TPB) are replaced by python string processing.
kernels ="""
__global__ void h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j>0 && k>0 ) hx[idx] -= 0.5*( ez[idx] - ez[idx-nz] - ey[idx] + ey[idx-1] );
	if( k>0 && i>0 ) hy[idx] -= 0.5*( ex[idx] - ex[idx-1] - ez[idx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( ey[idx] - ey[idx-nyz] - ex[idx] + ex[idx-nz] );
}

__global__ void e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( k<nz-1 && i<nx-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

//--------------------------------------------------------------------------------------------------------------------------
__global__ void h_aligned(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	__shared__ float s[2*TPB+2];
	float *sy = &s[1];
	float *sx = &sy[TPB+1];

	sy[tx] = ey[idx];
	sx[tx] = ex[idx];
	if( tx == 0 && k > 0 ) {
		sy[tx-1] = ey[idx-1];
		sx[tx-1] = ex[idx-1];
	}
	__syncthreads();

	if( j>0 ) hx[idx] -= 0.5*( ez[idx] - ez[idx-nz] - ey[idx] + sy[tx-1] );
	if( i>0 ) hy[idx] -= 0.5*( ex[idx] - sx[tx-1] - ez[idx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( ey[idx] - ey[idx-nyz] - ex[idx] + ex[idx-nz] );
}

__global__ void e_aligned(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	__shared__ float s[2*TPB+2];
	float *sy = s;
	float *sx = &sy[TPB+1];

	sy[tx] = hy[idx];
	sx[tx] = hx[idx];
	if( tx == TPB-1 && k < nz-1 ) {
		sy[tx+1] = hy[idx+1];
		sx[tx+1] = hx[idx+1];
	}
	__syncthreads();

	if( j<ny-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - sy[tx+1] + hy[idx] );
	if( i<nx-1 ) ey[idx] += cey[idx]*( sx[tx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

//--------------------------------------------------------------------------------------------------------------------------
__global__ void h_aligned_duplicated(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
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
	if( tx == 0 && k > 0 ) {
		sy[tx-1] = ey[idx-1];
		sx[tx-1] = ex[idx-1];
	}
	__syncthreads();

	if( j>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-nz] - sy[tx] + sy[tx-1] );
	if( i>0 ) hy[idx] -= 0.5*( sx[tx] - sx[tx-1] - sz[tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - ey[idx-nyz] - sx[tx] + ex[idx-nz] );
}

__global__ void e_aligned_duplicated(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	__shared__ float s[3*TPB+2];
	float *sz = s;
	float *sy = &sz[TPB];
	float *sx = &sy[TPB+1];

	sz[tx] = hz[idx];
	sy[tx] = hy[idx];
	sx[tx] = hx[idx];
	if( tx == TPB-1 && k < nz-1 ) {
		sy[tx+1] = hy[idx+1];
		sx[tx+1] = hx[idx+1];
	}
	__syncthreads();

	if( j<ny-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - sz[tx] - sy[tx+1] + sy[tx] );
	if( i<nx-1 ) ey[idx] += cey[idx]*( sx[tx+1] - sx[tx] - hz[idx+nyz] + sz[tx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - sy[tx] - hx[idx+nz] + sx[tx] );
}
"""
  
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

MAX_GRID = 65535

nx, ny, nz = 240, 256, 256
flop = nx*ny*nz*15

# measure kernel execution time
start = cuda.Event()
stop = cuda.Event()

navg = 100
exec_time = np.zeros(navg)


def measure_exec_time(str, func, *args):
	for i in xrange(navg):
		start.record()
		func(*args)
		stop.record()
		stop.synchronize()
		exec_time[i] = stop.time_since(start)
	t_mean = exec_time.mean()
	print '%s\tFLOPS: %1.2f GB/s (%1.5f, %1.5f)' % ( str, flop/t_mean*1e-6, t_mean, exec_time.std() )


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

	return cf


tpb = 256
bpg = (nx*ny*nz)/tpb

print 'dim (%d, %d, %d)' % (nx, ny, nz)
total_bytes = nx*ny*nz*4*9
if total_bytes/(1024**3) == 0:
	print 'mem %d MB' % ( total_bytes/(1024**2) )
else:
	print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

# memory allocate
f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz))
#f = np.zeros((nx,ny,nz), 'f')
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

# prepare kernels
mod = SourceModule( kernels.replace('TPB',str(tpb)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
h = mod.get_function("h")
e = mod.get_function("e")
h_aligned = mod.get_function("h_aligned")
e_aligned = mod.get_function("e_aligned")
h_aligned_duplicated = mod.get_function("h_aligned_duplicated")
e_aligned_duplicated = mod.get_function("e_aligned_duplicated")

h.prepare("PPPPPP", block=(tpb,1,1))
e.prepare("PPPPPPPPP", block=(tpb,1,1))
h_aligned.prepare("PPPPPP", block=(tpb,1,1))
e_aligned.prepare("PPPPPPPPP", block=(tpb,1,1))
h_aligned_duplicated.prepare("PPPPPP", block=(tpb,1,1))
e_aligned_duplicated.prepare("PPPPPPPPP", block=(tpb,1,1))

h_args = ((bpg,1), ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)
e_args = ((bpg,1), ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu)

measure_exec_time('h', h.prepared_call, *h_args)
measure_exec_time('e', e.prepared_call, *e_args)
measure_exec_time('h_aligned', h_aligned.prepared_call, *h_args)
measure_exec_time('e_aligned', e_aligned.prepared_call, *e_args)
measure_exec_time('h_aligned_duplicated', h_aligned_duplicated.prepared_call, *h_args)
measure_exec_time('e_aligned_duplicated', e_aligned_duplicated.prepared_call, *e_args)
