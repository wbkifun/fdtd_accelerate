#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535

kernels = """
texture<float, 1, cudaReadModeElementType> tcex;
texture<float, 1, cudaReadModeElementType> tcey;
texture<float, 1, cudaReadModeElementType> tcez;
texture<float, 1, cudaReadModeElementType> tchx;
texture<float, 1, cudaReadModeElementType> tchy;
texture<float, 1, cudaReadModeElementType> tchz;

__global__ void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x];
	float* sz = (float*) &sy[blockDim.x];

	sx[tx] = hx[idx];
	sy[tx] = hy[idx];
	sz[tx] = hz[idx];
	__syncthreads();

	float syp = sy[tx+1];
	float szp = sz[tx+1];
	if( tx == blockDim.x - 1 ) {
		syp = hy[idx+1];
		szp = hz[idx+1];
	}

	if( idx < (nx*ny-1)*nz ) 
		ex[idx] += tex1Dfetch(tcex,idx)*( hz[idx+nx] - sz[tx] - hy[idx+nx*ny] + sy[tx] );

	if( idx < nx*ny*(nz-1) ) {
		ey[idx] += tex1Dfetch(tcey,idx)*( hx[idx+nx*ny] - sx[tx] - szp + sz[tx] );
		ez[idx] += tex1Dfetch(tcez,idx)*( syp - sy[tx] - hx[idx+nx] + sx[tx] );
	}
}

__global__ void update_h(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x];
	float* sz = (float*) &sy[blockDim.x];

	sx[tx] = ex[idx];
	sy[tx] = ey[idx];
	sz[tx] = ez[idx];
	__syncthreads();

	float syp = sy[tx-1];
	float szp = sz[tx-1];
	if( tx == 0 ) {
		syp = ey[idx-1];
		szp = ez[idx-1];
	}

	if( idx < nx*ny*nz ) {
		if( idx > nz ) 
			hx[idx] -= tex1Dfetch(tchx,idx)*( sz[tx] - ez[idx-nx] - sy[tx] + ey[idx-ny*nz] );

		if( idx > ny*nz ) {
			hy[idx] -= tex1Dfetch(tchy,idx)*( sx[tx] - ex[idx-ny*nz] - sz[tx] + szp );
			hz[idx] -= tex1Dfetch(tchz,idx)*( sy[tx] - syp - sx[tx] + ex[idx-nx] );
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


if __name__ == '__main__':
	nx, ny, nz = 320, 320, 320

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	print 'mem %1.2f GB' % ( nx*ny*nz*4*12./(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz),'f',order='F')
	cf = np.zeros_like(f)

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device( set_c(cf,(None,-1,-1)) )
	cey_gpu = cuda.to_device( set_c(cf,(-1,None,-1)) )
	cez_gpu = cuda.to_device( set_c(cf,(-1,-1,None)) )
	chx_gpu = cuda.to_device( set_c(cf,(None,0,0)) )
	chy_gpu = cuda.to_device( set_c(cf,(0,None,0)) )
	chz_gpu = cuda.to_device( set_c(cf,(0,0,None)) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_e = mod.get_function("update_e")
	update_h = mod.get_function("update_h")
	update_src = mod.get_function("update_src")
	tcex = mod.get_texref("tcex")
	tcey = mod.get_texref("tcey")
	tcez = mod.get_texref("tcez")
	tchx = mod.get_texref("tchx")
	tchy = mod.get_texref("tchy")
	tchz = mod.get_texref("tchz")

	tcex.set_address(cex_gpu, cf.nbytes)
	tcey.set_address(cey_gpu, cf.nbytes)
	tcez.set_address(cez_gpu, cf.nbytes)
	tchx.set_address(chx_gpu, cf.nbytes)
	tchy.set_address(chy_gpu, cf.nbytes)
	tchz.set_address(chz_gpu, cf.nbytes)

	tpb = 512
	bpg = (nx*ny*nz)/tpb

	Db = (tpb,1,1)
	Dg = (bpg,1)

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_e.prepare("iiiPPPPPP", block=Db, texrefs=[tcex,tcey,tcez], shared=tpb*3*4)
	update_h.prepare("iiiPPPPPP", block=Db, texrefs=[tchx,tchy,tchz], shared=tpb*3*4)
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
		update_e.prepared_call(Dg, nnx, nny, nnz, ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

		update_h.prepared_call(Dg, nnx, nny, nnz, ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)
		
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
