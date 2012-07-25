#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
texture<float, 1, cudaReadModeElementType> tex;
texture<float, 1, cudaReadModeElementType> tey;
texture<float, 1, cudaReadModeElementType> tez;
texture<float, 1, cudaReadModeElementType> thx;
texture<float, 1, cudaReadModeElementType> thy;
texture<float, 1, cudaReadModeElementType> thz;
texture<float, 3, cudaReadModeElementType> tcex;
texture<float, 3, cudaReadModeElementType> tcey;
texture<float, 3, cudaReadModeElementType> tcez;
texture<float, 3, cudaReadModeElementType> tchx;
texture<float, 3, cudaReadModeElementType> tchy;
texture<float, 3, cudaReadModeElementType> tchz;

__global__ void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez) {
	int by = blockIdx.y%(ny/blockDim.y);
	int bz = blockIdx.y/(ny/blockDim.y);
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = by*blockDim.y + threadIdx.y;
	int k = bz*blockDim.z + threadIdx.z;
	int idx = k*nx*ny + j*nx + i;

	if( idx < (nx*ny-1)*nz ) 
		ex[idx] += tex3D(tcex,i,j,k)*( tex1Dfetch(thz,idx+nx) - tex1Dfetch(thz,idx) - tex1Dfetch(thy,idx+nx*ny) + tex1Dfetch(thy,idx) );

	if( idx < nx*ny*(nz-1) ) {
		ey[idx] += tex3D(tcey,i,j,k)*( tex1Dfetch(thx,idx+nx*ny) - tex1Dfetch(thx,idx) - tex1Dfetch(thz,idx+1) + tex1Dfetch(thz,idx) );
		ez[idx] += tex3D(tcez,i,j,k)*( tex1Dfetch(thy,idx+1) - tex1Dfetch(thy,idx) - tex1Dfetch(thx,idx+nx) + tex1Dfetch(thx,idx) );
	}
}

__global__ void update_h(int nx, int ny, int nz, float *hx, float *hy, float *hz) {
	int by = blockIdx.y%(ny/blockDim.y);
	int bz = blockIdx.y/(ny/blockDim.y);
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = by*blockDim.y + threadIdx.y;
	int k = bz*blockDim.z + threadIdx.z;
	int idx = k*nx*ny + j*nx + i;

	if( idx < nx*ny*nz ) {
		if( idx > nz ) 
			hx[idx] -= tex3D(tchx,i,j,k)*( tex1Dfetch(tez,idx) - tex1Dfetch(tez,idx-nx) - tex1Dfetch(tey,idx) + tex1Dfetch(tey,idx-nx*ny) );

		if( idx > ny*nz ) {
			hy[idx] -= tex3D(tchy,i,j,k)*( tex1Dfetch(tex,idx) - tex1Dfetch(tex,idx-nx*ny) - tex1Dfetch(tez,idx) + tex1Dfetch(tez,idx-1) );
			hz[idx] -= tex3D(tchz,i,j,k)*( tex1Dfetch(tey,idx) - tex1Dfetch(tey,idx-1) - tex1Dfetch(tex,idx) + tex1Dfetch(tex,idx-nx) );
		}
	}
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
	nx, ny, nz = 320, 320, 320

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	print 'mem %1.2f GB' % ( nx*ny*nz*4*12./(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz),'f',order='F')

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

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

	memcopy(mcopy, set_c(f,(None,-1,-1)), cex_gpu)
	memcopy(mcopy, set_c(f,(-1,None,-1)), cey_gpu)
	memcopy(mcopy, set_c(f,(-1,-1,None)), cez_gpu)
	memcopy(mcopy, set_c(f,(None,0,0)), chx_gpu)
	memcopy(mcopy, set_c(f,(0,None,0)), chy_gpu)
	memcopy(mcopy, set_c(f,(0,0,None)), chz_gpu)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_e = mod.get_function("update_e")
	update_h = mod.get_function("update_h")
	update_src = mod.get_function("update_src")
	tex = mod.get_texref("tex")
	tey = mod.get_texref("tey")
	tez = mod.get_texref("tez")
	thx = mod.get_texref("thx")
	thy = mod.get_texref("thy")
	thz = mod.get_texref("thz")
	tcex = mod.get_texref("tcex")
	tcey = mod.get_texref("tcey")
	tcez = mod.get_texref("tcez")
	tchx = mod.get_texref("tchx")
	tchy = mod.get_texref("tchy")
	tchz = mod.get_texref("tchz")

	tex.set_address(ex_gpu, f.nbytes)
	tey.set_address(ey_gpu, f.nbytes)
	tez.set_address(ez_gpu, f.nbytes)
	thx.set_address(hx_gpu, f.nbytes)
	thy.set_address(hy_gpu, f.nbytes)
	thz.set_address(hz_gpu, f.nbytes)
	tcex.set_array(cex_gpu)
	tcey.set_array(cey_gpu)
	tcez.set_array(cez_gpu)
	tchx.set_array(chx_gpu)
	tchy.set_array(chy_gpu)
	tchz.set_array(chz_gpu)


	Db = (16,8,4)
	Dg = (nx/16, ny*nz/(8*4))

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_e.prepare("iiiPPP", block=Db, texrefs=[thx,thy,thz,tcex,tcey,tcez])
	update_h.prepare("iiiPPP", block=Db, texrefs=[tex,tey,tez,tchx,tchy,tchz])
	update_src.prepare("iiiiP", block=(512,1,1))

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((nx,ny),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.001)
	colorbar()

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()
	
	# main loop
	for tn in xrange(1, 101):
		update_e.prepared_call(Dg, nnx, nny, nnz, ex_gpu, ey_gpu, ez_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

		update_h.prepared_call(Dg, nnx, nny, nnz, hx_gpu, hy_gpu, hz_gpu)
		
		'''
		if tn%100 == 0:
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
