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


kernels = """
texture<float, 3, cudaReadModeElementType> tcf;

__global__ void naive(float *f, float *cf) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	f[idx] = cf[idx];
}

__global__ void tex3d(float *f) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	f[idx] = tex3D(tcf,k,j,i);
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

	# memory allocate
	f = np.zeros((nx,ny,nz), 'f')
	g = np.zeros((nx,ny,nz), 'f')
	cf = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz))

	f_gpu = cuda.to_device(f)
	g_gpu = cuda.to_device(f)
	cf_gpu = cuda.to_device( cf )

	descr = cuda.ArrayDescriptor3D()
	descr.width = nz
	descr.height = ny
	descr.depth = nx
	descr.format = cuda.dtype_to_array_format(f.dtype)
	descr.num_channels = 1
	descr.flags = 0
	tcf_gpu = cuda.Array(descr)

	mcopy = cuda.Memcpy3D()
	mcopy.width_in_bytes = mcopy.src_pitch = f.strides[1]
	mcopy.src_height = mcopy.height = ny
	mcopy.depth = nx
	arrcopy(mcopy, cf, tcf_gpu)

	# prepare kernels
	from pycuda.compiler import SourceModule
	kernels = kernels.replace('Dx','16').replace('Dy','16').replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz))
	mod = SourceModule( kernels )

	naive = mod.get_function("naive")
	tex3d = mod.get_function("tex3d")

	tcf = mod.get_texref("tcf")
	tcf.set_array(tcf_gpu)

	# measure kernel execution time
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	naive( f_gpu, cf_gpu, block=(16,16,1), grid=(nz/16,nx*ny/16) )
	tex3d( g_gpu, block=(16,16,1), grid=(nz/16,nx*ny/16), texrefs=[tcf] )
	cuda.memcpy_dtoh(f, f_gpu)
	cuda.memcpy_dtoh(g, g_gpu)
	assert( np.linalg.norm(f-g) == 0 )

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
