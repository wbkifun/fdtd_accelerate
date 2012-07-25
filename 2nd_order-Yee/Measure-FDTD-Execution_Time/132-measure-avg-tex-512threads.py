#!/usr/bin/env python

# Constant variables (nx, ny, nz, nyz, TPB) are replaced by python string processing.
kernels1 ="""
__global__ void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j>0 && k>0 ) hx[idx] -= 0.5*( ez[idx] - ez[idx-nz] - ey[idx] + ey[idx-1] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( ex[idx] - ex[idx-1] - ez[idx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( ey[idx] - ey[idx-nyz] - ex[idx] + ex[idx-nz] );
}

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}
"""

kernels2 ="""
__global__ void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
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

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
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
"""

kernels3 ="""
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
	if( tx == 0 && k > 0 ) {
		sy[tx-1] = ey[idx-1];
		sx[tx-1] = ex[idx-1];
	}
	__syncthreads();

	if( j>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-nz] - sy[tx] + sy[tx-1] );
	if( i>0 ) hy[idx] -= 0.5*( sx[tx] - sx[tx-1] - sz[tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - ey[idx-nyz] - sx[tx] + ex[idx-nz] );
}

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
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

kernels4 ="""
__global__ void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty+1][tx+1] = ex[idx];
	sy[ty][tx+1] = ey[idx];
	sz[ty+1][tx] = ez[idx];
	if( tx == 0 && k > 0 ) {
		sx[ty+1][0] = ex[idx-1];
		sy[ty][0] = ey[idx-1];
	}
	if( ty == 0 && j > 0 ) {
		sx[0][tx+1] = ex[idx-nz];
		sz[0][tx] = ez[idx-nz];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[ty+1][tx+1] - sx[ty+1][tx] - sz[ty+1][tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[ty][tx+1] - ey[idx-nyz] - sx[ty+1][tx+1] + sx[ty][tx+1] );
}

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty][tx] = hx[idx];
	sy[ty][tx] = hy[idx];
	sz[ty][tx] = hz[idx];
	if( tx == Dx-1 && k < nz-1 ) {
		sx[ty][Dx] = hx[idx+1];
		sy[ty][Dx] = hy[idx+1];
	}
	if( ty == Dy-1 && j < ny-1 ) {
		sx[Dy][tx] = hx[idx+nz];
		sz[Dy][tx] = hz[idx+nz];
	}
	__syncthreads();

	ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i<nx-1 ) {
		ey[idx] += cey[idx]*( sx[ty][tx+1] - sx[ty][tx] - hz[idx+nyz] + sz[ty][tx] );
		ez[idx] += cez[idx]*( hy[idx+nyz] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
	}
}
"""

kernels5 ="""
texture<float, 3, cudaReadModeElementType> tcex;
texture<float, 3, cudaReadModeElementType> tcey;
texture<float, 3, cudaReadModeElementType> tcez;

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty][tx] = hx[idx];
	sy[ty][tx] = hy[idx];
	sz[ty][tx] = hz[idx];
	if( tx == Dx-1 && k < nz-1 ) {
		sx[ty][Dx] = hx[idx+1];
		sy[ty][Dx] = hy[idx+1];
	}
	if( ty == Dy-1 && j < ny-1 ) {
		sx[Dy][tx] = hx[idx+nz];
		sz[Dy][tx] = hz[idx+nz];
	}
	__syncthreads();

	ex[idx] += tex3D(tcex,k,j,i)*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i<nx-1 ) {
		ey[idx] += tex3D(tcey,k,j,i)*( sx[ty][tx+1] - sx[ty][tx] - hz[idx+nyz] + sz[ty][tx] );
		ez[idx] += tex3D(tcez,k,j,i)*( hy[idx+nyz] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
	}
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
	print '%1.2f GFLOPS (%1.5f, %1.5f)\t%s' % ( flop/t_mean*1e-6, t_mean, exec_time.std(), str )


def set_c(f, direction):
	f[:,:,:] = 0.5
	if 'x' in direction: f[-1,:,:] = 0
	if 'y' in direction: f[:,-1,:] = 0
	if 'z' in direction: f[:,:,-1] = 0

	return f


def arrcopy(mcopy, src, dst):
	mcopy.set_src_host( src )
	mcopy.set_dst_array( dst )
	mcopy()


Dx, Dy = 32, 16
tpb = Dx*Dy
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

ex_gpu = cuda.to_device(f)
ey_gpu = cuda.to_device(f)
ez_gpu = cuda.to_device(f)
hx_gpu = cuda.to_device(f)
hy_gpu = cuda.to_device(f)
hz_gpu = cuda.to_device(f)

cex = set_c(f,'yz')
cey = set_c(f,'zx')
cez = set_c(f,'xy')

cex_gpu = cuda.to_device( cex )
cey_gpu = cuda.to_device( cey )
cez_gpu = cuda.to_device( cez )

descr = cuda.ArrayDescriptor3D()
descr.width = nz
descr.height = ny
descr.depth = nx
descr.format = cuda.dtype_to_array_format(f.dtype)
descr.num_channels = 1
descr.flags = 0
tcex_gpu = cuda.Array(descr)
tcey_gpu = cuda.Array(descr)
tcez_gpu = cuda.Array(descr)

mcopy = cuda.Memcpy3D()
mcopy.width_in_bytes = mcopy.src_pitch = f.strides[1]
mcopy.src_height = mcopy.height = ny
mcopy.depth = nx
arrcopy(mcopy, cex, tcex_gpu)
arrcopy(mcopy, cey, tcey_gpu)
arrcopy(mcopy, cez, tcez_gpu)

# prepare kernels
mod1 = SourceModule( kernels1.replace('TPB',str(tpb)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
mod2 = SourceModule( kernels2.replace('TPB',str(tpb)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
mod3 = SourceModule( kernels3.replace('TPB',str(tpb)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
mod4 = SourceModule( kernels4.replace('Dx',str(Dx)).replace('Dy',str(Dy)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
mod5 = SourceModule( kernels5.replace('Dx',str(Dx)).replace('Dy',str(Dy)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )

h1 = mod1.get_function("update_h")
e1 = mod1.get_function("update_e")
h2 = mod2.get_function("update_h")	# avoid mis-aligned
e2 = mod2.get_function("update_e")
h3 = mod3.get_function("update_h")	# avoid mis-aligned, duplicated
e3 = mod3.get_function("update_e")
h4 = mod4.get_function("update_h")	# avoid mis-aligned, duplicated and 2d block
e4 = mod4.get_function("update_e")
e5 = mod5.get_function("update_e")	# avoid mis-aligned, duplicated and 2d block, tex3D

tcex = mod5.get_texref("tcex")
tcey = mod5.get_texref("tcey")
tcez = mod5.get_texref("tcez")
tcex.set_array(tcex_gpu)
tcey.set_array(tcey_gpu)
tcez.set_array(tcez_gpu)

h1.prepare("PPPPPP", block=(tpb,1,1))
e1.prepare("PPPPPPPPP", block=(tpb,1,1))
h2.prepare("PPPPPP", block=(tpb,1,1))
e2.prepare("PPPPPPPPP", block=(tpb,1,1))
h3.prepare("PPPPPP", block=(tpb,1,1))
e3.prepare("PPPPPPPPP", block=(tpb,1,1))
h4.prepare("PPPPPP", block=(Dx,Dy,1))
e4.prepare("PPPPPPPPP", block=(Dx,Dy,1))
e5.prepare("PPPPPP", block=(Dx,Dy,1), texrefs=[tcex,tcey,tcez])

eh_args = (ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)
ehc_args = (ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu)

print '-'*47
print 'update_e: Measure FLOPS (time mean, time std)'
print '-'*47
measure_exec_time('naive', e1.prepared_call, (bpg,1), *ehc_args)
measure_exec_time('mis-aligned', e2.prepared_call, (bpg,1), *ehc_args)
measure_exec_time('duplicated', e3.prepared_call, (bpg,1), *ehc_args)
measure_exec_time('2d block', e4.prepared_call, (nz/Dx,nx*ny/Dy), *ehc_args)
measure_exec_time('tex3D (32,16,1)', e5.prepared_call, (nz/Dx,nx*ny/Dy), *eh_args)

print '\n','-'*47
print 'update_h: Measure FLOPS (time mean, time std)'
print '-'*47
measure_exec_time('naive', h1.prepared_call, (bpg,1), *eh_args)
measure_exec_time('mis-aligned', h2.prepared_call, (bpg,1), *eh_args)
measure_exec_time('duplicated', h3.prepared_call, (bpg,1), *eh_args)
measure_exec_time('2d block', h4.prepared_call, (nz/Dx,nx*ny/Dy), *eh_args)
print '-'*47
