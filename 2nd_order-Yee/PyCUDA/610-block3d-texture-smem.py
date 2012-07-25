#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
texture<float, 3, cudaReadModeElementType> tcex;
texture<float, 3, cudaReadModeElementType> tcey;
texture<float, 3, cudaReadModeElementType> tcez;
texture<float, 3, cudaReadModeElementType> tchx;
texture<float, 3, cudaReadModeElementType> tchy;
texture<float, 3, cudaReadModeElementType> tchz;

__global__ void update_e(int nx, int ny, int nz, int gDy, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int Dx = blockDim.x;
	int Dy = blockDim.y;
	int Dz = blockDim.z;
	int sidx = tz*Dx*Dy + ty*Dx + tx;

	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int i = blockIdx.x*Dx + tx;
	int j = by*Dy + ty;
	int k = bz*Dz + tz;
	int idx = k*nx*ny + j*nx +i;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[Dx*Dy*Dz];
	float* sz = (float*) &sy[Dx*Dy*Dz];

	sx[sidx] = hx[idx];
	sy[sidx] = hy[idx];
	sz[sidx] = hz[idx];
	__syncthreads();

	float hy_x = sy[sidx+1];
	float hz_x = sz[sidx+1];
	float hx_y = sx[sidx+Dx];
	float hz_y = sz[sidx+Dx];
	float hx_z = sx[sidx+Dx*Dy];
	float hy_z = sy[sidx+Dx*Dy];
	if( tx == Dx-1 ) {
		hy_x = hy[idx+1];
		hz_x = hz[idx+1];
	}
	if( ty == Dy-1 ) {
		hx_y = hx[idx+nx];
		hz_y = hz[idx+nx];
	}
	if( tz == Dz-1 ) {
		hx_z = hx[idx+nx*ny];
		hy_z = hy[idx+nx*ny];
	}

	if( idx < nx*ny*(nz-1) ) {
		ex[idx] += tex3D(tcex,i,j,k)*( hz_y - sz[sidx] - hy_z + sy[sidx] );
		ey[idx] += tex3D(tcey,i,j,k)*( hx_z - sx[sidx] - hz_x + sz[sidx] );
	}

	if( idx < nx*(ny*nz-1) )
		ez[idx] += tex3D(tcez,i,j,k)*( hy_x - sy[sidx] - hx_y + sx[sidx] );
}

__global__ void update_h(int nx, int ny, int nz, int gDy, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int Dx = blockDim.x;
	int Dy = blockDim.y;
	int Dz = blockDim.z;
	int sidx = tz*Dx*Dy + ty*Dx + tx;

	int by = blockIdx.y%gDy;
	int bz = blockIdx.y/gDy;
	int i = blockIdx.x*Dx + tx;
	int j = by*Dy + ty;
	int k = bz*Dz + tz;
	int idx = k*nx*ny + j*nx +i;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[Dx*Dy*Dz];
	float* sz = (float*) &sy[Dx*Dy*Dz];

	sx[sidx] = ex[idx];
	sy[sidx] = ey[idx];
	sz[sidx] = ez[idx];
	__syncthreads();

	float ey_x = sy[sidx-1];
	float ez_x = sz[sidx-1];
	float ex_y = sx[sidx-Dx];
	float ez_y = sz[sidx-Dx];
	float ex_z = sx[sidx-Dx*Dy];
	float ey_z = sy[sidx-Dx*Dy];
	if( tx == 0 ) {
		ey_x = ey[idx-1];
		ez_x = ez[idx-1];
	}
	if( ty == 0 ) {
		ex_y = ex[idx-nx];
		ez_y = ez[idx-nx];
	}
	if( tz == 0 ) {
		ex_z = ex[idx-nx*ny];
		ey_z = ey[idx-nx*ny];
	}

	if( idx > nx*ny ) {
		hx[idx] -= tex3D(tchx,i,j,k)*( sz[sidx] - ez_y - sy[sidx] + ey_z );
		hy[idx] -= tex3D(tchx,i,j,k)*( sx[sidx] - ex_z - sz[sidx] + ez_x );
	}
	
	if( idx > nx ) 
		hz[idx] -= tex3D(tchx,i,j,k)*( sy[sidx] - ey_x - sx[sidx] + ex_y );
}

__global__ void update_src(int nx, int ny, int nz, int tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nz/2)*nx*ny + (ny/2)*nx + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
"""


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

	return cf


def arrcopy(mcopy, src, dst):
	mcopy.set_src_host( src )
	mcopy.set_dst_array( dst )
	mcopy()


if __name__ == '__main__':
	nx, ny, nz = 256, 256, 240

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*12
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

	# memory allocate with cuda array
	descr = cuda.ArrayDescriptor3D()
	descr.width = nx
	descr.height = ny
	descr.depth = nz
	descr.format = cuda.dtype_to_array_format(f.dtype)
	descr.num_channels = 1
	descr.flags = 0

	cex_gpu = cuda.Array(descr)
	cey_gpu = cuda.Array(descr)
	cez_gpu = cuda.Array(descr)
	chx_gpu = cuda.Array(descr)
	chy_gpu = cuda.Array(descr)
	chz_gpu = cuda.Array(descr)

	mcopy = cuda.Memcpy3D()
	mcopy.width_in_bytes = mcopy.src_pitch = f.strides[1]
	mcopy.src_height = mcopy.height = ny
	mcopy.depth = nz

	arrcopy(mcopy, set_c(f,(None,-1,-1)), cex_gpu)
	arrcopy(mcopy, set_c(f,(-1,None,-1)), cey_gpu)
	arrcopy(mcopy, set_c(f,(-1,-1,None)), cez_gpu)
	arrcopy(mcopy, set_c(f,(None,0,0)), chx_gpu)
	arrcopy(mcopy, set_c(f,(0,None,0)), chy_gpu)
	arrcopy(mcopy, set_c(f,(0,0,None)), chz_gpu)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_e = mod.get_function("update_e")
	update_h = mod.get_function("update_h")
	update_src = mod.get_function("update_src")

	# bind a texture reference to linear memory
	tcex = mod.get_texref("tcex")
	tcey = mod.get_texref("tcey")
	tcez = mod.get_texref("tcez")
	tchx = mod.get_texref("tchx")
	tchy = mod.get_texref("tchy")
	tchz = mod.get_texref("tchz")

	tcex.set_array(cex_gpu)
	tcey.set_array(cey_gpu)
	tcez.set_array(cez_gpu)
	tchx.set_array(chx_gpu)
	tchy.set_array(chy_gpu)
	tchz.set_array(chz_gpu)


	Db = (16,4,4)
	Dg = (nx/16, ny*nz/(4*4))
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	gDy = np.int32(ny/Db[1])
	update_e.prepare("iiiiPPPPPP", block=Db, texrefs=[tcex,tcey,tcez], shared=16*4*4*3*4)
	update_h.prepare("iiiiPPPPPP", block=Db, texrefs=[tchx,tchy,tchz], shared=16*4*4*3*4)
	update_src.prepare("iiiiP", block=(256,1,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.001)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	# main loop
	for tn in xrange(1, 1001):
		update_e.prepared_call(
				Dg, nnx, nny, nnz, gDy,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

		update_h.prepared_call(
				Dg, nnx, nny, nnz, gDy,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)
			
		'''
		if tn%100 == 0:
		#if tn == 100:
			print 'tn =', tn
			cuda.memcpy_dtoh(f, ex_gpu)
			imsh.set_array( f[nx/2,:,:]**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
		'''

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
