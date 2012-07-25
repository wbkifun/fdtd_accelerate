#!/usr/bin/env python

# Access Patterns used in 3D FDTD
#
#
# f[idx], f[idx+1], f[idx+nx], f[idx+nxy]		# row-major (C)
#
# if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
# if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+ny*nz] + hz[idx] );
# if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+ny*nz] - hy[idx] - hx[idx+nz] + hx[idx] );
#
#
# f[idx], f[idx+1], f[idx+nz], f[idx+nyz]		# column-major (Fortran)
#
# if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nx] - hz[idx] - hy[idx+nxy] + hy[idx] );
# if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - hx[idx] - hz[idx+1] + hz[idx] );
# if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+1] - hy[idx] - hx[idx+nx] + hx[idx] );
#
# 
# 
# along the thread block dimensions 
# 
# 1D block
# 
# 
# 2D block
# 
#
# 3D block
#
#
#
#
# texture fetch with linear array
#
#
#
#
#
#
#
# texture fetch with CUDA array
#
#
#
#
#
# successive access effect
#
#


kernels = """
#define TPB 256
#define Dz 16
#define Dy 4
#define Dx 4

texture<float, 3, cudaReadModeElementType> tcex;
texture<float, 3, cudaReadModeElementType> tcey;
texture<float, 3, cudaReadModeElementType> tcez;

__global__ void e_naive(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

__global__ void e_naive_shuffle(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float rcx = cex[idx];
	float rcy = cey[idx];
	float rcz = cez[idx];
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] += rcx*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += rcy*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += rcz*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

__global__ void e_naive_shuffle2(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float rex = ex[idx];
	float rey = ey[idx];
	float rez = ez[idx];
	float rcx = cex[idx];
	float rcy = cey[idx];
	float rcz = cez[idx];
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] = rex + rcx*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] = rey + rcy*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] = rez + rcz*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
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
	if( tx == TPB-1 && k<nz-1) {
		sy[tx+1] = hy[idx+1];
		sx[tx+1] = hx[idx+1];
	}
	__syncthreads();

	if( j<ny-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - sy[tx+1] + hy[idx] );
	if( i<nx-1 ) ey[idx] += cey[idx]*( sx[tx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

__global__ void e_reuse(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
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
	if( tx == TPB-1 && k<nz-1 ) {
		sy[tx+1] = hy[idx+1];
		sx[tx+1] = hx[idx+1];
	}
	__syncthreads();

	if( j<ny-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - sz[tx] - sy[tx+1] + sy[tx] );
	if( i<nx-1 ) ey[idx] += cey[idx]*( sx[tx+1] - sx[tx] - hz[idx+nyz] + sz[tx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - sy[tx] - hx[idx+nz] + sx[tx] );
}

__global__ void e_tex(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] += tex3D(tcex,k,j,i)*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += tex3D(tcey,k,j,i)*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += tex3D(tcez,k,j,i)*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

__global__ void e_3d_naive(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tz = threadIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.z;
	int bz = blockIdx.x%(nz/Dz);
	int by = blockIdx.x/(nz/Dz);
	int bx = blockIdx.y;
	int k = bz*Dz + tz;
	int j = by*Dy + ty;
	int i = bx*Dx + tx;
	int idx = i*nyz + j*nz + k;

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

__global__ void e_3d_reuse(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tz = threadIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.z;
	int bz = blockIdx.x%(nz/Dz);
	int by = blockIdx.x/(nz/Dz);
	int bx = blockIdx.y;
	int k = bz*Dz + tz;
	int j = by*Dy + ty;
	int i = bx*Dx + tx;
	int idx = i*nyz + j*nz + k;
	int sidz = tx*Dz*(Dy+1) + ty*Dz + tz;
	int sidy = tx*(Dz+1)*Dy + ty*(Dz+1) + tz;
	int sidx = tx*(Dz+1)*(Dy+1) + ty*(Dz+1) + tz;

	__shared__ float sz[(Dx*Dy+Dx+Dy)*Dz];
	__shared__ float sy[(Dz*Dx+Dz+Dx)*Dy];
	__shared__ float sx[(Dy*Dz+Dy+Dz)*Dx];

	sz[sidz] = hz[idx];
	sy[sidy] = hy[idx];
	sx[sidx] = hx[idx];
	if( tz == Dz-1 && k<nz-1 ) {
		sy[sidy+1] = hy[idx+1];
		sx[sidx+1] = hx[idx+1];
	}
	if( ty == Dy-1 && j<ny-1 ) {
		sz[sidz+Dz] = hz[idx+nz];
		sx[sidx+Dz+1] = hx[idx+nz];
	}
	if( tx == Dx-1 && i<nx-1 ) {
		sz[sidz+Dz*(Dy+1)] = hz[idx+nyz];
		sy[sidy+(Dz+1)*Dy] = hy[idx+nyz];
	}
	__syncthreads();
		
	ex[idx] += cex[idx]*( sz[sidz+Dz] - sz[sidz] - sy[sidy+1] + sy[sidy] );
	ey[idx] += cey[idx]*( sx[sidx+1] - sx[sidx] - sz[sidz+Dz*(Dy+1)] + sz[sidz] );
	ez[idx] += cez[idx]*( sy[sidy+(Dz+1)*Dy] - sy[sidy] - sx[sidx+Dz+1] + sx[sidx] );
}
"""

import numpy as np
import sys
import pycuda.driver as cuda
import pycuda.autoinit


def arrcopy(mcopy, src, dst):
	mcopy.set_src_host( src )
	mcopy.set_dst_array( dst )
	mcopy()

if __name__ == '__main__':
	nx, ny, nz = 240, 256, 256

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	#f = np.zeros((nx,ny,nz), 'f')
	f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz))
	cf = np.ones_like(f)*0.5

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device( cf )
	cey_gpu = cuda.to_device( cf )
	cez_gpu = cuda.to_device( cf )

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

	arrcopy(mcopy, cf, tcex_gpu)
	arrcopy(mcopy, cf, tcey_gpu)
	arrcopy(mcopy, cf, tcez_gpu)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule( kernels.replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
	e_naive = mod.get_function("e_naive")
	e_naive_shuffle = mod.get_function("e_naive_shuffle")
	e_naive_shuffle2 = mod.get_function("e_naive_shuffle2")
	e_aligned = mod.get_function("e_aligned")
	e_reuse = mod.get_function("e_reuse")
	e_tex = mod.get_function("e_tex")
	e_3d_naive = mod.get_function("e_3d_naive")
	e_3d_reuse = mod.get_function("e_3d_reuse")

	tcex = mod.get_texref("tcex")
	tcey = mod.get_texref("tcey")
	tcez = mod.get_texref("tcez")
	tcex.set_array(tcex_gpu)
	tcey.set_array(tcey_gpu)
	tcez.set_array(tcez_gpu)

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	e_naive( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1) )
	e_naive_shuffle( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1) )
	e_naive_shuffle2( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1) )
	e_aligned( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1) )
	e_reuse( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1) )
	e_tex( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1), texrefs=[tcex,tcey,tcez])

	#e_3d_naive( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(16,4,4), grid=(nx*ny/(16*4),nz/4) )
	#e_3d_reuse( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(16,4,4), grid=(nx*ny/(16*4),nz/4) )

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
