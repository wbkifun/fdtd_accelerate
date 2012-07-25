#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels =""" 
#define Dx2 32
#define Dy2 16
#define Dx3 16
#define Dy3 8
#define Dz3 4

__global__ void update_1d(int nx, int ny, int nz, float *a, float *b, float *c, float *d, float *e, float *f, float *g) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	a[idx] += b[idx] - c[idx] + d[idx] -e[idx] + f[idx] - g[idx];
}

__global__ void update_2d(int nx, int ny, int nz, float *a, float *b, float *c, float *d, float *e, float *f, float *g) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = blockIdx.x*Dx2 + tx;
	int gj = blockIdx.y*Dy2 + ty;
	int j = gj%ny;
	int k = gj/ny;
	int idx = i + gj*nx;

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


if __name__ == '__main__':
	nx, ny, nz = 256, 256, 240

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

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)

	Db1 = (512,1,1)
	Dg1 = ((nx*ny*nz)/512, 1)
	Db2 = (32,16,1)
	Dg2 = (nx/32, (ny*nz)/16)
	Db3 = (16,8,4)
	Dg3 = (nx/16, (ny*nz)/(8*4))
	update_1d.prepare("iiiPPPPPPP", block=Db1)
	update_2d.prepare("iiiPPPPPPP", block=Db2)
	update_3d.prepare("iiiPPPPPPP", block=Db3)

	start = cuda.Event()
	stop = cuda.Event()
	start.record()
	for tn in xrange(1, 1000+1):
		update_1d.prepared_call(Dg1, nnx, nny, nnz, a1_gpu, b_gpu, c_gpu, d_gpu, e_gpu, f_gpu, g_gpu)
	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	cuda.memcpy_dtoh(a1, a1_gpu)

	start.record()
	for tn in xrange(1, 1000+1):
		update_2d.prepared_call(Dg2, nnx, nny, nnz, a2_gpu, b_gpu, c_gpu, d_gpu, e_gpu, f_gpu, g_gpu)
	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	cuda.memcpy_dtoh(a2, a2_gpu)

	start.record()
	for tn in xrange(1, 1000+1):
		update_3d.prepared_call(Dg3, nnx, nny, nnz, a3_gpu, b_gpu, c_gpu, d_gpu, e_gpu, f_gpu, g_gpu)
	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	cuda.memcpy_dtoh(a3, a3_gpu)

	assert la.norm(a1-a2) == 0
	assert la.norm(a2-a3) == 0
	assert la.norm(a3-a2) == 0
