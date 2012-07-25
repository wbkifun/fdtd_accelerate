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
#define Dx 32
#define Dy 16

texture<float, 1, cudaReadModeElementType> thx;
texture<float, 1, cudaReadModeElementType> thy;
texture<float, 1, cudaReadModeElementType> thz;

__global__ void e_naive(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int i = idx%nx;
	int j = (idx%nxy)/nx;
	int k = idx/nxy;

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nx] - hz[idx] - hy[idx+nxy] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - hx[idx] - hz[idx+1] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+1] - hy[idx] - hx[idx+nx] + hx[idx] );
}

__global__ void e_reuse(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx;
	int i = idx%nx;
	int j = (idx%nxy)/nx;
	int k = idx/nxy;

	__shared__ float s[3*TPB+2];
	float *sx = s;
	float *sy = &sx[TPB];
	float *sz = &sy[TPB+1];

	sx[tx] = hx[idx];
	sy[tx] = hy[idx];
	sz[tx] = hz[idx];
	if( tx == TPB-1 && i<nx-1 ) {
		sy[TPB] = hy[idx+1];
		sz[TPB] = hz[idx+1];
	}
	__syncthreads();

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nx] - sz[tx] - hy[idx+nxy] + sy[tx] );
	if( k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - sx[tx] - sz[tx+1] + sz[tx] );
	if( j<ny-1 ) ez[idx] += cez[idx]*( sy[tx+1] - sy[tx] - hx[idx+nx] + sx[tx] );
}

__global__ void e_reuse_tex(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx;
	int i = idx%nx;
	int j = (idx%nxy)/nx;
	int k = idx/nxy;

	__shared__ float s[3*TPB+2];
	float *sx = s;
	float *sy = &sx[TPB];
	float *sz = &sy[TPB+1];

	sx[tx] = tex1Dfetch(thx,idx);
	sy[tx] = tex1Dfetch(thy,idx);
	sz[tx] = tex1Dfetch(thz,idx);
	if( tx == TPB-1 && i<nx-1 ) {
		sy[TPB] = hy[idx+1];
		sz[TPB] = hz[idx+1];
	}
	__syncthreads();

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nx] - sz[tx] - hy[idx+nxy] + sy[tx] );
	if( k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - sx[tx] - sz[tx+1] + sz[tx] );
	if( j<ny-1 ) ez[idx] += cez[idx]*( sy[tx+1] - sy[tx] - hx[idx+nx] + sx[tx] );
}

__global__ void e_reuse_2dblk(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int i = bx*blockDim.x + tx;
	int j = (by*blockDim.y + ty)%ny;
	int k = (by*blockDim.y + ty)/ny;
	int idx = k*nxy + j*nx + i;

	__shared__ float sx[Dy+1][Dx];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx+1];

	sx[ty][tx] = hx[idx];
	sy[ty][tx] = hy[idx];
	sz[ty][tx] = hz[idx];
	if( tx == Dx-1 && i<nx-1 ) {
		sy[ty][Dx] = hy[idx+1];
		sz[ty][Dx] = hz[idx+1];
	}
	if( ty == Dy-1 && j<ny-1 ) {
		sx[Dy][tx] = hx[idx+nx];
		sz[Dy][tx] = hz[idx+nx];
	}

	if( k<nz-1 ) ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - hy[idx+nxy] + sy[ty][tx] );
	if( k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - sx[ty][tx] - sz[ty][tx+1] + sz[ty][tx] );
	ez[idx] += cez[idx]*( sy[ty][tx+1] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
}

__global__ void e_reuse_2dblk_tex(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int i = bx*blockDim.x + tx;
	int j = (by*blockDim.y + ty)%ny;
	int k = (by*blockDim.y + ty)/ny;
	int idx = k*nxy + j*nx + i;

	__shared__ float sx[Dy+1][Dx];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx+1];

	sx[ty][tx] = tex1Dfetch(thx,idx);
	sy[ty][tx] = tex1Dfetch(thy,idx);
	sz[ty][tx] = tex1Dfetch(thz,idx);
	if( tx == Dx-1 && i<nx-1 ) {
		sy[ty][Dx] = hy[idx+1];
		sz[ty][Dx] = hz[idx+1];
	}
	if( ty == Dy-1 && j<ny-1 ) {
		sx[Dy][tx] = hx[idx+nx];
		sz[Dy][tx] = hz[idx+nx];
	}

	if( k<nz-1 ) ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - hy[idx+nxy] + sy[ty][tx] );
	if( k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - sx[ty][tx] - sz[ty][tx+1] + sz[ty][tx] );
	ez[idx] += cez[idx]*( sy[ty][tx+1] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
}

__global__ void e_reuse_2dblk_tex2(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int i = bx*blockDim.x + tx;
	int j = (by*blockDim.y + ty)%ny;
	int k = (by*blockDim.y + ty)/ny;
	int idx = k*nxy + j*nx + i;

	__shared__ float sx[Dy+1][Dx];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx+1];

	sx[ty][tx] = tex1Dfetch(thx,idx);
	sy[ty][tx] = tex1Dfetch(thy,idx);
	sz[ty][tx] = tex1Dfetch(thz,idx);
	if( tx == Dx-1 && i<nx-1 ) {
		sy[ty][Dx] = tex1Dfetch(thy,idx+1);
		sz[ty][Dx] = tex1Dfetch(thz,idx+1);
	}
	if( ty == Dy-1 && j<ny-1 ) {
		sx[Dy][tx] = tex1Dfetch(thx,idx+nx);
		sz[Dy][tx] = tex1Dfetch(thz,idx+nx);
	}

	if( k<nz-1 ) ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - hy[idx+nxy] + sy[ty][tx] );
	if( k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+nxy] - sx[ty][tx] - sz[ty][tx+1] + sz[ty][tx] );
	ez[idx] += cez[idx]*( sy[ty][tx+1] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
}

__global__ void e_reuse_2dblk_tex3(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int i = bx*blockDim.x + tx;
	int j = (by*blockDim.y + ty)%ny;
	int k = (by*blockDim.y + ty)/ny;
	int idx = k*nxy + j*nx + i;

	__shared__ float sx[Dy+1][Dx];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx+1];

	sx[ty][tx] = tex1Dfetch(thx,idx);
	sy[ty][tx] = tex1Dfetch(thy,idx);
	sz[ty][tx] = tex1Dfetch(thz,idx);
	if( tx == Dx-1 && i<nx-1 ) {
		sy[ty][Dx] = tex1Dfetch(thy,idx+1);
		sz[ty][Dx] = tex1Dfetch(thz,idx+1);
	}
	if( ty == Dy-1 && j<ny-1 ) {
		sx[Dy][tx] = tex1Dfetch(thx,idx+nx);
		sz[Dy][tx] = tex1Dfetch(thz,idx+nx);
	}

	if( k<nz-1 ) ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - tex1Dfetch(thy,idx+nxy) + sy[ty][tx] );
	if( k<nz-1 ) ey[idx] += cey[idx]*( tex1Dfetch(thx,idx+nxy) - sx[ty][tx] - sz[ty][tx+1] + sz[ty][tx] );
	ez[idx] += cez[idx]*( sy[ty][tx+1] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
}

__global__ void e_reuse_2dblk_tex4(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int i = bx*blockDim.x + tx;
	int j = (by*blockDim.y + ty)%ny;
	int k = (by*blockDim.y + ty)/ny;
	int idx = k*nxy + j*nx + i;

	__shared__ float sx[Dy+1][Dx];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx+1];

	sx[ty][tx] = hx[idx];
	sy[ty][tx] = hy[idx];
	sz[ty][tx] = hz[idx];
	if( tx == Dx-1 && i<nx-1 ) {
		sy[ty][Dx] = hy[idx+1];
		sz[ty][Dx] = hz[idx+1];
	}
	if( ty == Dy-1 && j<ny-1 ) {
		sx[Dy][tx] = hx[idx+nx];
		sz[Dy][tx] = hz[idx+nx];
	}

	if( k<nz-1 ) ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - tex1Dfetch(thy,idx+nxy) + sy[ty][tx] );
	if( k<nz-1 ) ey[idx] += cey[idx]*( tex1Dfetch(thx,idx+nxy) - sx[ty][tx] - sz[ty][tx+1] + sz[ty][tx] );
	ez[idx] += cez[idx]*( sy[ty][tx+1] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
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
	#f = np.zeros((nx,ny,nz), 'f', order='F')
	f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz), order='F')
	cf = np.ones((nx,ny,nz), 'f', order='F')*0.5

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device( cf )
	cey_gpu = cuda.to_device( cf )
	cez_gpu = cuda.to_device( cf )

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule( kernels.replace('nxy',str(nx*ny)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
	e_naive = mod.get_function("e_naive")
	e_reuse = mod.get_function("e_reuse")
	e_reuse_tex = mod.get_function("e_reuse_tex")
	e_reuse_2dblk = mod.get_function("e_reuse_2dblk")
	e_reuse_2dblk_tex = mod.get_function("e_reuse_2dblk_tex")
	e_reuse_2dblk_tex2 = mod.get_function("e_reuse_2dblk_tex2")
	e_reuse_2dblk_tex3 = mod.get_function("e_reuse_2dblk_tex3")
	e_reuse_2dblk_tex4 = mod.get_function("e_reuse_2dblk_tex4")

	thx = mod.get_texref("thx")
	thy = mod.get_texref("thy")
	thz = mod.get_texref("thz")
	thx.set_address(hx_gpu, f.nbytes)
	thy.set_address(hy_gpu, f.nbytes)
	thz.set_address(hz_gpu, f.nbytes)

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	e_naive( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1) )
	e_reuse( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1) )
	e_reuse_tex( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(256,1,1), grid=(nx*ny*nz/256,1), texrefs=[thx,thy,thz])
	e_reuse_2dblk( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(32,16,1), grid=(nx/32,ny*nz/16))
	e_reuse_2dblk_tex( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(32,16,1), grid=(nx/32,ny*nz/16), texrefs=[thx,thy,thz])
	e_reuse_2dblk_tex2( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(32,16,1), grid=(nx/32,ny*nz/16), texrefs=[thx,thy,thz])
	e_reuse_2dblk_tex3( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(32,16,1), grid=(nx/32,ny*nz/16), texrefs=[thx,thy,thz])
	e_reuse_2dblk_tex4( ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, cex_gpu, cey_gpu, cez_gpu, block=(32,16,1), grid=(nx/32,ny*nz/16), texrefs=[thx,thy,thz])

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
