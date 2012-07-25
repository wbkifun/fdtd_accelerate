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
texture<float, 1, cudaReadModeElementType> tcex;
texture<float, 1, cudaReadModeElementType> tcey;
texture<float, 1, cudaReadModeElementType> tcez;
texture<float, 1, cudaReadModeElementType> tchx;
texture<float, 1, cudaReadModeElementType> tchy;
texture<float, 1, cudaReadModeElementType> tchz;

__global__ void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if( idx < (nx*ny-1)*nz ) 
		ex[idx] += tex1Dfetch(tcex,idx)*( tex1Dfetch(thz,idx+nz) - tex1Dfetch(thz,idx) - tex1Dfetch(thy,idx+1) + tex1Dfetch(thy,idx) );

	if( idx < nx*ny*(nz-1) ) {
		ey[idx] += tex1Dfetch(tcey,idx)*( tex1Dfetch(thx,idx+1) - tex1Dfetch(thx,idx) - tex1Dfetch(thz,idx+ny*nz) + tex1Dfetch(thz,idx) );
		ez[idx] += tex1Dfetch(tcez,idx)*( tex1Dfetch(thy,idx+ny*nz) - tex1Dfetch(thy,idx) - tex1Dfetch(thx,idx+nz) + tex1Dfetch(thx,idx) );
	}
}

__global__ void update_h(int nx, int ny, int nz, float *hx, float *hy, float *hz) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if( idx < nx*ny*nz ) {
		if( idx > nz ) 
			hx[idx] -= tex1Dfetch(tchx,idx)*( tex1Dfetch(tez,idx) - tex1Dfetch(tez,idx-nz) - tex1Dfetch(tey,idx) + tex1Dfetch(tey,idx-1) );

		if( idx > ny*nz ) {
			hy[idx] -= tex1Dfetch(tchy,idx)*( tex1Dfetch(tex,idx) - tex1Dfetch(tex,idx-1) - tex1Dfetch(tez,idx) + tex1Dfetch(tez,idx-ny*nz) );
			hz[idx] -= tex1Dfetch(tchz,idx)*( tex1Dfetch(tey,idx) - tex1Dfetch(tey,idx-ny*nz) - tex1Dfetch(tex,idx) + tex1Dfetch(tex,idx-nz) );
		}
	}
}

__global__ void update_src(int nx, int ny, int nz, int tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nx/2)*ny*nz + (ny/2)*nz + idx;

	if( idx < nz ) f[ijk] += sin(0.1*tn);
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
	f = np.zeros((nx,ny,nz),'f')
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

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ez_gpu)

		update_h.prepared_call(Dg, nnx, nny, nnz, hx_gpu, hy_gpu, hz_gpu)
		
		'''
		if tn%100 == 0:
			print 'tn =', tn
			cuda.memcpy_dtoh(f, ez_gpu)
			imsh.set_array( f[:,:,nz/2]**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
		'''

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
