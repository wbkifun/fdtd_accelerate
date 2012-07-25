#!/usr/bin/env python

# Constant variables (nx, ny, nz, nyz, TPB) are replaced by python string processing.
kernels ="""
texture<float, 1, cudaReadModeElementType> thx;
texture<float, 1, cudaReadModeElementType> thy;
texture<float, 1, cudaReadModeElementType> thz;
texture<float, 1, cudaReadModeElementType> tcex;
texture<float, 1, cudaReadModeElementType> tcey;
texture<float, 1, cudaReadModeElementType> tcez;

__global__ void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	__shared__ float s[3*TPB+2];
	float *sz = s;
	float *sy = &sz[TPB+1];
	float *sx = &sy[TPB+1];

	sz[tx] = ez[idx];
	sy[tx] = ey[idx];
	sx[tx] = ex[idx];
	if( tx == 0 ) {
		sy[tx-1] = ey[idx-1];
		sx[tx-1] = ex[idx-1];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-nz] - sy[tx] + sy[tx-1] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[tx] - sx[tx-1] - sz[tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - ey[idx-nyz] - sx[tx] + ex[idx-nz] );
}

__global__ void update_e(float *ex, float *ey, float *ez) {
	int tx = threadIdx.x;
	int idx = 4*TPB*blockIdx.x + tx;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	__shared__ float s[15*TPB];
	float tmp_ezx[4], tmp_ey[4];
	float tmp_ce[4], tmp_h[4];

	//------------------------------------------------------------------------------
	// first thread block
	//------------------------------------------------------------------------------
	s[tx] = tex1Dfetch(thx,idx);
	s[tx+TPB] = tex1Dfetch(thx,idx+TPB);
	s[tx+2*TPB] = tex1Dfetch(thx,idx+2*TPB);
	s[tx+3*TPB] = tex1Dfetch(thx,idx+3*TPB);
	s[tx+4*TPB] = tex1Dfetch(thx,idx+4*TPB);
	__syncthreads();
	tmp_ezx[0] = s[tx+TPB] - s[tx];
	tmp_ezx[1] = s[tx+2*TPB] - s[tx+TPB];
	tmp_ezx[2] = s[tx+3*TPB] - s[tx+2*TPB];
	tmp_ezx[3] = s[tx+4*TPB] - s[tx+3*TPB];
	tmp_ey[0] = s[tx+1] - s[tx];
	tmp_ey[1] = s[tx+TPB+1] - s[tx+TPB];
	tmp_ey[2] = s[tx+2*TPB+1] - s[tx+2*TPB];
	tmp_ey[3] = s[tx+3*TPB+1] - s[tx+3*TPB];

	s[tx+5*TPB] = tex1Dfetch(thy,idx);
	s[tx+6*TPB] = tex1Dfetch(thy,idx+TPB);
	s[tx+7*TPB] = tex1Dfetch(thy,idx+2*TPB);
	s[tx+8*TPB] = tex1Dfetch(thy,idx+3*TPB);
	s[tx+9*TPB] = tex1Dfetch(thy,idx+4*TPB);
	__syncthreads();
	if( i<nx-1 && j<ny-1 ) {
		tmp_ce[0] = tex1Dfetch(tcez,idx);
		tmp_ce[1] = tex1Dfetch(tcez,idx+TPB);
		tmp_ce[2] = tex1Dfetch(tcez,idx+2*TPB);
		tmp_ce[3] = tex1Dfetch(tcez,idx+3*TPB);
		tmp_h[0] = tex1Dfetch(thy,idx+nyz);
		tmp_h[1] = tex1Dfetch(thy,idx+TPB+nyz);
		tmp_h[2] = tex1Dfetch(thy,idx+2*TPB+nyz);
		tmp_h[3] = tex1Dfetch(thy,idx+3*TPB+nyz);

		ez[idx] += tmp_ce[0]*( tmp_h[0] - s[tx+5*TPB] - tmp_ezx[0] );
		ez[idx+TPB] += tmp_ce[1]*( tmp_h[1] - s[tx+6*TPB] - tmp_ezx[1] );
		ez[idx+2*TPB] += tmp_ce[2]*( tmp_h[2] - s[tx+7*TPB] - tmp_ezx[2] );
		ez[idx+3*TPB] += tmp_ce[3]*( tmp_h[3] - s[tx+8*TPB] - tmp_ezx[3] );
	}
	tmp_ezx[0] = s[tx+5*TPB+1] - s[tx+5*TPB];
	tmp_ezx[1] = s[tx+6*TPB+1] - s[tx+6*TPB];
	tmp_ezx[2] = s[tx+7*TPB+1] - s[tx+7*TPB];
	tmp_ezx[3] = s[tx+8*TPB+1] - s[tx+8*TPB];

	s[tx+10*TPB] = tex1Dfetch(thz,idx);
	s[tx+11*TPB] = tex1Dfetch(thz,idx+TPB);
	s[tx+12*TPB] = tex1Dfetch(thz,idx+2*TPB);
	s[tx+13*TPB] = tex1Dfetch(thz,idx+3*TPB);
	s[tx+14*TPB] = tex1Dfetch(thz,idx+4*TPB);
	__syncthreads();
	if( i<nx-1 && k<nz-1 ) {
		tmp_ce[0] = tex1Dfetch(tcey,idx);
		tmp_ce[1] = tex1Dfetch(tcey,idx+TPB);
		tmp_ce[2] = tex1Dfetch(tcey,idx+2*TPB);
		tmp_ce[3] = tex1Dfetch(tcey,idx+3*TPB);
		tmp_h[0] = tex1Dfetch(thz,idx+nyz);
		tmp_h[1] = tex1Dfetch(thz,idx+TPB+nyz);
		tmp_h[2] = tex1Dfetch(thz,idx+2*TPB+nyz);
		tmp_h[3] = tex1Dfetch(thz,idx+3*TPB+nyz);

		ey[idx] += tmp_ce[0]*( tmp_ey[0] - tmp_h[0] + s[tx+10*TPB] );
		ey[idx+TPB] += tmp_ce[1]*( tmp_ey[1] - tmp_h[1] + s[tx+11*TPB] );
		ey[idx+2*TPB] += tmp_ce[2]*( tmp_ey[2] - tmp_h[2] + s[tx+12*TPB] );
		ey[idx+3*TPB] += tmp_ce[3]*( tmp_ey[3] - tmp_h[3] + s[tx+13*TPB] );
	}
	if( j<ny-1 && k<nz-1 ) {
		tmp_ce[0] = tex1Dfetch(tcex,idx);
		tmp_ce[1] = tex1Dfetch(tcex,idx+TPB);
		tmp_ce[2] = tex1Dfetch(tcex,idx+2*TPB);
		tmp_ce[3] = tex1Dfetch(tcex,idx+3*TPB);

		ex[idx] += tmp_ce[0]*( s[tx+11*TPB] - s[tx+10*TPB] - tmp_ezx[0] );
		ex[idx+TPB] += tmp_ce[1]*( s[tx+12*TPB] - s[tx+11*TPB] - tmp_ezx[1] );
		ex[idx+2*TPB] += tmp_ce[2]*( s[tx+13*TPB] - s[tx+12*TPB] - tmp_ezx[2] );
		ex[idx+3*TPB] += tmp_ce[3]*( s[tx+14*TPB] - s[tx+13*TPB] - tmp_ezx[3] );
	}
}

__global__ void update_src(float tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nx/2)*ny*nz + (ny/2)*nz + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}
"""
  
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

MAX_GRID = 65535


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

	return cf



if __name__ == '__main__':
	nx, ny, nz = 240, 256, 256
	tpb = 256
	bpg = (nx*ny*nz)/tpb

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*9
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), 'f')
	cf = np.ones_like(f)*0.5

	ex_gpu = cuda.to_device(f)
	ey_gpu = cuda.to_device(f)
	ez_gpu = cuda.to_device(f)
	hx_gpu = cuda.to_device(f)
	hy_gpu = cuda.to_device(f)
	hz_gpu = cuda.to_device(f)

	cex_gpu = cuda.to_device( set_c(cf,(None,-1,-1)) )
	cey_gpu = cuda.to_device( set_c(cf,(-1,None,-1)) )
	cez_gpu = cuda.to_device( set_c(cf,(-1,-1,None)) )

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule( kernels.replace('TPB',str(tpb)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
	update_h = mod.get_function("update_h")
	update_e = mod.get_function("update_e")
	update_src = mod.get_function("update_src")
	thx = mod.get_texref("thx")
	thy = mod.get_texref("thy")
	thz = mod.get_texref("thz")
	tcex = mod.get_texref("tcex")
	tcey = mod.get_texref("tcey")
	tcez = mod.get_texref("tcez")

	thx.set_address(hx_gpu, f.nbytes)
	thy.set_address(hy_gpu, f.nbytes)
	thz.set_address(hz_gpu, f.nbytes)
	tcex.set_address(cex_gpu, cf.nbytes)
	tcey.set_address(cey_gpu, cf.nbytes)
	tcez.set_address(cez_gpu, cf.nbytes)

	update_h.prepare("PPPPPP", block=(tpb,1,1), texrefs=[thx,thy,thz,tcex,tcey,tcez])
	update_e.prepare("PPP", block=(tpb,1,1))
	update_src.prepare("fP", block=(nz,1,1))

	'''
	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((nx,ny),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
	colorbar()
	'''

	# measure kernel execution time
	#from datetime import datetime
	#t1 = datetime.now()
	start = cuda.Event()
	stop = cuda.Event()
	start.record()

	# main loop
	for tn in xrange(1, 1000+1):
		update_h.prepared_call(
				(bpg,1), ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		update_e.prepared_call(
				(bpg/4,1), ex_gpu, ey_gpu, ez_gpu)

		update_src.prepared_call((1,1), np.float32(tn), ez_gpu)

		'''
		if tn%10 == 0:
		#if tn == 100:
			print 'tn =', tn
			cuda.memcpy_dtoh(f, ez_gpu)
			imsh.set_array( f[:,:,nz/2].T**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
		'''

	stop.record()
	stop.synchronize()
	print stop.time_since(start)*1e-3
	#print datetime.now() - t1
