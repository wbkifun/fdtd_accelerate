#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535
start = cuda.Event()
stop = cuda.Event()

nx, ny, nz = 256, 256, 240
nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)

kernels =""" 
#define Dx1 512
#define Dy1 1
#define Dz1 1

#define Dx2 32
#define Dy2 16
#define Dz2 1

#define Dx3 32
#define Dy3 4
#define Dz3 4

__global__ void update_1d(int nx, int ny, int nz, float *a, float *b, float *c, float *d, float *e, float *f, float *g) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int gDy = ny/Dy1;
	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int i = blockIdx.x*Dx1+tx;
	int j = by*Dy1+ty;
	int k = bz*Dz1+tz;
	int idx = i + j*nx + k*nx*ny;

	a[idx] += b[idx] - c[idx] + d[idx] -e[idx] + f[idx] - g[idx];
}

__global__ void update_2d(int nx, int ny, int nz, float *a, float *b, float *c, float *d, float *e, float *f, float *g) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int gDy = ny/Dy2;
	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int i = blockIdx.x*Dx2+tx;
	int j = by*Dy2+ty;
	int k = bz*Dz2+tz;
	int idx = i + j*nx + k*nx*ny;

	a[idx] += b[idx] - c[idx] + d[idx] -e[idx] + f[idx] - g[idx];
}

__global__ void update_3d(int nx, int ny, int nz, float *a, float *b, float *c, float *d, float *e, float *f, float *g) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int gDy = ny/Dy3;
	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int i = blockIdx.x*Dx3+tx;
	int j = by*Dy3+ty;
	int k = bz*Dz3+tz;
	int idx = i + j*nx + k*nx*ny;

	a[idx] += b[idx] - c[idx] + d[idx] -e[idx] + f[idx] - g[idx];
}
"""

def time_check(kern, Dg, Db, a_gpu):
	start.record()
	for tn in xrange(1, 1000+1):
		kern(nnx, nny, nnz, a_gpu, b_gpu, c_gpu, d_gpu, e_gpu, f_gpu, g_gpu,grid=Dg, block=Db)
	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3


if __name__ == '__main__':

	#f = np.zeros((nx,ny,nz), 'f', order='F')
	f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz),order='F')
	a1_gpu = cuda.to_device(np.zeros_like(f))
	a2_gpu = cuda.to_device(np.zeros_like(f))
	a3_gpu = cuda.to_device(np.zeros_like(f))
	b_gpu = cuda.to_device(f*0.1)
	c_gpu = cuda.to_device(f*0.2)
	d_gpu = cuda.to_device(f*0.3)
	e_gpu = cuda.to_device(f*0.4)
	f_gpu = cuda.to_device(f*0.5)
	g_gpu = cuda.to_device(f*0.6)

	a1 = np.zeros_like(f)
	a2 = np.zeros_like(f)
	a3 = np.zeros_like(f)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_1d = mod.get_function("update_1d")
	update_2d = mod.get_function("update_2d")
	update_3d = mod.get_function("update_3d")

	time_check(update_1d, ((nx*ny*nz)/512, 1), (512,1,1), a1_gpu)
	time_check(update_2d, (nx/32, (ny*nz)/16), (32,16,1), a2_gpu)
	time_check(update_3d, (nx/32, (ny*nz)/(4*4)), (32,4,4), a3_gpu)

	cuda.memcpy_dtoh(a1, a1_gpu)
	cuda.memcpy_dtoh(a2, a2_gpu)
	cuda.memcpy_dtoh(a3, a3_gpu)
	assert la.norm(a1-a2) == 0
	assert la.norm(a2-a3) == 0
