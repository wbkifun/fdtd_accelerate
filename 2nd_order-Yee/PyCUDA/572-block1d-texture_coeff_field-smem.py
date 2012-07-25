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

texture<float, 1, cudaReadModeElementType> tex;
texture<float, 1, cudaReadModeElementType> tey;
texture<float, 1, cudaReadModeElementType> tez;
texture<float, 1, cudaReadModeElementType> thx;
texture<float, 1, cudaReadModeElementType> thy;
texture<float, 1, cudaReadModeElementType> thz;

__global__ void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x+1];
	float* sz = (float*) &sy[blockDim.x+1];

	sx[tx] = tex1Dfetch(thx,idx);
	sy[tx] = tex1Dfetch(thy,idx);
	sz[tx] = tex1Dfetch(thz,idx);
	__syncthreads();

	float hy_x = sy[tx+1];
	float hz_x = sz[tx+1];
	if( tx == blockDim.x - 1 ) {
		hy_x = tex1Dfetch(thy,idx+1);
		hz_x = tex1Dfetch(thz,idx+1);
	}

	if( idx < nx*ny*(nz-1) ) {
		ex[idx] += tex1Dfetch(tcex,idx)*( tex1Dfetch(thz,idx+nx) - sz[tx] - tex1Dfetch(thy,idx+nx*ny) + sy[tx] );
		ey[idx] += tex1Dfetch(tcey,idx)*( tex1Dfetch(thx,idx+nx*ny) - sx[tx] - hz_x + sz[tx] );
	}

	if( idx < nx*(ny*nz-1) )
		ez[idx] += tex1Dfetch(tcez,idx)*( hy_x - sy[tx] - tex1Dfetch(thx,idx+nx) + sx[tx] );
}

__global__ void update_h(int nx, int ny, int nz, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x];
	float* sz = (float*) &sy[blockDim.x];

	sx[tx] = tex1Dfetch(tex,idx);
	sy[tx] = tex1Dfetch(tey,idx);
	sz[tx] = tex1Dfetch(tez,idx);
	__syncthreads();

	float ey_x = sy[tx-1];
	float ez_x = sz[tx-1];
	if( tx == 0 ) {
		ey_x = tex1Dfetch(tey,idx-1);
		ez_x = tex1Dfetch(tez,idx-1);
	}

	if( idx > nx*ny ) {
		hx[idx] -= tex1Dfetch(tchx,idx)*( sz[tx] - tex1Dfetch(tez,idx-nx) - sy[tx] + tex1Dfetch(tey,idx-nx*ny) );
		hy[idx] -= tex1Dfetch(tchy,idx)*( sx[tx] - tex1Dfetch(tex,idx-nx*ny) - sz[tx] + ez_x );
	}
	
	if( idx > nx ) 
		hz[idx] -= tex1Dfetch(tchz,idx)*( sy[tx] - ey_x - sx[tx] + tex1Dfetch(tex,idx-nx) );
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

	cex_gpu = cuda.to_device( set_c(f,(None,-1,-1)) )
	cey_gpu = cuda.to_device( set_c(f,(-1,None,-1)) )
	cez_gpu = cuda.to_device( set_c(f,(-1,-1,None)) )
	chx_gpu = cuda.to_device( set_c(f,(None,0,0)) )
	chy_gpu = cuda.to_device( set_c(f,(0,None,0)) )
	chz_gpu = cuda.to_device( set_c(f,(0,0,None)) )

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

	tex = mod.get_texref("tex")
	tey = mod.get_texref("tey")
	tez = mod.get_texref("tez")
	thx = mod.get_texref("thx")
	thy = mod.get_texref("thy")
	thz = mod.get_texref("thz")

	tcex.set_address(cex_gpu, f.nbytes)
	tcey.set_address(cey_gpu, f.nbytes)
	tcez.set_address(cez_gpu, f.nbytes)
	tchx.set_address(chx_gpu, f.nbytes)
	tchy.set_address(chy_gpu, f.nbytes)
	tchz.set_address(chz_gpu, f.nbytes)

	tex.set_address(ex_gpu, f.nbytes)
	tey.set_address(ey_gpu, f.nbytes)
	tez.set_address(ez_gpu, f.nbytes)
	thx.set_address(hx_gpu, f.nbytes)
	thy.set_address(hy_gpu, f.nbytes)
	thz.set_address(hz_gpu, f.nbytes)

	tpb = 256
	bpg = (nx*ny*nz)/tpb

	Db = (tpb,1,1)
	Dg = (bpg, 1)
	print Db, Dg

	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
	update_e.prepare("iiiPPP", block=Db, texrefs=[tcex,tcey,tcez,thx,thy,thz], shared=tpb*3*4)
	update_h.prepare("iiiPPP", block=Db, texrefs=[tchx,tchy,tchz,tex,tey,tez], shared=tpb*3*4)
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
		update_e.prepared_call(Dg, nnx, nny, nnz, ex_gpu, ey_gpu, ez_gpu)

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

		update_h.prepared_call(Dg, nnx, nny, nnz, hx_gpu, hy_gpu, hz_gpu)
			
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