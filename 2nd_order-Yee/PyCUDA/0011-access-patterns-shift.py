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
__global__ void func00(float *fw) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float f=3.1415;
	__syncthreads();
	fw[idx] = f;
}

__global__ void func01(float *fw, float *fx) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float f;
	f = fx[idx];
	__syncthreads();
	fw[idx] = f;
}

__global__ void func02(float *fw, float *fx, float *fy) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float f,g;
	f = fx[idx];
	g = fy[idx];
	__syncthreads();
	fw[idx] = f+g;
}

__global__ void func03(float *fw, float *fx, float *fy) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float f,g;
	f = fx[idx];
	__syncthreads();
	g = fy[idx];
	__syncthreads();
	fw[idx] = f+g;
}

__global__ void func04(float *fw, float *fx, float *fy, float *fz) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float x,y,z;
	x = fx[idx];
	y = fy[idx];
	z = fz[idx];
	__syncthreads();
	fw[idx] = x+y+z;
}

__global__ void func05(float *fw, float *fx, float *fy, float *fz) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	float x,y,z;
	x = fx[idx];
	__syncthreads();
	y = fy[idx];
	__syncthreads();
	z = fz[idx];
	__syncthreads();
	fw[idx] = x+y+z;
}

#define Dx 256 

__global__ void func06(float *fw, float *fx, float *fy, float *fz) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx;
	__shared__ float sx[Dx];
	__shared__ float sy[Dx];
	__shared__ float sz[Dx];
	sx[tx] = fx[idx];
	sy[tx] = fy[idx];
	sz[tx] = fz[idx];
	__syncthreads();
	fw[idx] = sx[tx]+sy[tx]+sz[tx];
}

__global__ void func07(float *fw, float *fx, float *fy, float *fz) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx;
	__shared__ float sx[Dx];
	__shared__ float sy[Dx];
	__shared__ float sz[Dx];
	sx[tx] = fx[idx];
	__syncthreads();
	sy[tx] = fy[idx];
	__syncthreads();
	sz[tx] = fz[idx];
	__syncthreads();
	fw[idx] = sx[tx]+sy[tx]+sz[tx];
}
"""

import numpy as np
import sys
import pycuda.driver as cuda
import pycuda.autoinit


if __name__ == '__main__':
	nx, ny, nz = 240, 256, 256

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz))
	fw_gpu = cuda.to_device( np.zeros((nx,ny,nz), 'f') )
	fx_gpu = cuda.to_device(f)
	fy_gpu = cuda.to_device(f)
	fz_gpu = cuda.to_device(f)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule( kernels.replace('N',str((nx-1)*ny*nz)).replace('nz',str(nz)).replace('nyz',str(ny*nz)) )
	func00 = mod.get_function("func00")
	func01 = mod.get_function("func01")
	func02 = mod.get_function("func02")
	func03 = mod.get_function("func03")
	func04 = mod.get_function("func04")
	func05 = mod.get_function("func05")
	func06 = mod.get_function("func06")
	func07 = mod.get_function("func07")

	tpb = (256,1,1)
	bpg = (nx*ny*nz/256,1)

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	func00(fw_gpu, block=tpb, grid=bpg)
	func01(fw_gpu, fx_gpu, block=tpb, grid=bpg)
	func02(fw_gpu, fx_gpu, fy_gpu, block=tpb, grid=bpg)
	func03(fw_gpu, fx_gpu, fy_gpu, block=tpb, grid=bpg)
	func04(fw_gpu, fx_gpu, fy_gpu, fz_gpu, block=tpb, grid=bpg)
	func05(fw_gpu, fx_gpu, fy_gpu, fz_gpu, block=tpb, grid=bpg)
	func06(fw_gpu, fx_gpu, fy_gpu, fz_gpu, block=tpb, grid=bpg)
	func07(fw_gpu, fx_gpu, fy_gpu, fz_gpu, block=tpb, grid=bpg)

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
