#!/usr/bin/env python

# Constant variables (nx, ny, nz, nyz, nxyz, Dx) are replaced by python string processing.
kernels ="""
__global__ void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = Dx*blockIdx.x + tx;

	__shared__ float s[3*Dx+2];
	float *sz = s;
	float *sy = &sz[Dx+1];
	float *sx = &sy[Dx+1];

	while( idx < nxyz ) {
		sz[tx] = ez[idx];
		sy[tx] = ey[idx];
		sx[tx] = ex[idx];
		if( tx == 0 ) {
			sy[tx-1] = ey[idx-1];
			sx[tx-1] = ex[idx-1];
		}
		__syncthreads();

		int i = idx/(nyz);
		int j = (idx - i*nyz)/nz;
		int k = idx%nz;

		if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-nz] - sy[tx] + sy[tx-1] );
		if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[tx] - sx[tx-1] - sz[tx] + ez[idx-nyz] );
		if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - ey[idx-nyz] - sx[tx] + ex[idx-nz] );

		idx += Dx * gridDim.x;
	}
}

__global__ void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = Dx*blockIdx.x + tx;
	
	__shared__ float s[3*Dx+2];
	float *sz = s;
	float *sy = &sz[Dx];
	float *sx = &sy[Dx+1];

	while( idx < nxyz ) {
		sz[tx] = hz[idx];
		sy[tx] = hy[idx];
		sx[tx] = hx[idx];
		if( tx == Dx-1 ) {
			sy[tx+1] = hy[idx+1];
			sx[tx+1] = hx[idx+1];
		}
		__syncthreads();

		int i = idx/(nyz);
		int j = (idx - i*nyz)/nz;
		int k = idx%nz;

		if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - sz[tx] - sy[tx+1] + sy[tx] );
		if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( sx[tx+1] - sx[tx] - hz[idx+nyz] + sz[tx] );
		if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - sy[tx] - hx[idx+nz] + sx[tx] );

		idx += Dx * gridDim.x;
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
import sys

cuda.init()
MAX_BLOCK = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
MAX_MEM = 4056	# MByte
ngpu = cuda.Device.count()
ctx = cuda.Device(0).make_context()


nx, ny, nz = 240, 256, 256
tmax, tgap = 1000, 100

print '(%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx*ny*nz*4*9
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

if nz%32 != 0:
	print "Error: nz is not multiple of 32"
	sys.exit()


# memory allocate
f = np.zeros((nx,ny,nz), 'f')
cf = np.ones_like(f)*0.5

eh_gpus = ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu = [cuda.to_device(f) for i in range(6)]
ce_gpus = cex_gpu, cey_gpu, cez_gpu = [cuda.to_device(cf) for i in range(3)]


# prepare kernels
tpb = 256
for bpg in xrange(65535, 0, -1):
	if (nx * ny * nz / tpb) % bpg == 0: break
print 'tpb = %d, bpg = %g' % (tpb, bpg)

from pycuda.compiler import SourceModule
mod = SourceModule( kernels.replace('Dx',str(tpb)).replace('nxyz',str(nx*ny*nz)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)), options=['-m 64'] )
update_h = mod.get_function("update_h")
update_e = mod.get_function("update_e")
update_src = mod.get_function("update_src")

update_h.prepare("PPPPPP", block=(tpb,1,1))
update_e.prepare("PPPPPPPPP", block=(tpb,1,1))
update_src.prepare("fP", block=(nz,1,1))


# prepare for plot
#import matplotlib.pyplot as plt
#plt.ion()
#imsh = plt.imshow(np.ones((nx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
#plt.colorbar()


# measure kernel execution time
from datetime import datetime
t1 = datetime.now()
flop = (nx*ny*nz*30)*tgap
flops = np.zeros(tmax/tgap+1)
start, stop = cuda.Event(), cuda.Event()
start.record()

start2, stop2 = cuda.Event(), cuda.Event()
exec_time = {'update_h':np.zeros(tmax), 'update_e':np.zeros(tmax), 'src_e':np.zeros(tmax)}


# main loop
for tn in xrange(1, tmax+1):
	start2.record()
	update_h.prepared_call((bpg, 1), *eh_gpus)
	stop2.record()
	stop2.synchronize()
	exec_time['update_h'][tn-1] = stop2.time_since(start2)
	start2.record()

	update_e.prepared_call((bpg, 1), *(eh_gpus + ce_gpus))
	stop2.record()
	stop2.synchronize()
	exec_time['update_e'][tn-1] = stop2.time_since(start2)
	start2.record()

	update_src.prepared_call((1, 1), np.float32(tn), ez_gpu)
	stop2.record()
	stop2.synchronize()
	exec_time['src_e'][tn-1] = stop2.time_since(start2)


	if tn%tgap == 0:
		stop.record()
		stop.synchronize()
		flops[tn/tgap] = flop/stop.time_since(start)*1e-6
		print '[',datetime.now()-t1,']'," %d/%d (%d %%) %1.2f GFLOPS\r" % (tn, tmax, float(tn)/tmax*100, flops[tn/tgap]),
		sys.stdout.flush()
		#cuda.memcpy_dtoh(f, ez_gpu)
		#imsh.set_array( f[:,:,nz/2].T**2 )
		#plt.draw()
		start.record()

print "\navg: %1.6f GFLOPS" % flops[2:-2].mean()

total = np.zeros(tmax)
for key in exec_time.iterkeys(): total[:] += exec_time[key][:]
for key in exec_time.iterkeys():
	print key, ':\t %1.6f %%' % ( exec_time[key][2:-2].sum()/total[2:-2].sum()*100 )

print "%1.6f GFLOPS\r" % ( (tmax-4)*nx*ny*nz*30/total[2:-2].sum()*1e-6 )

ctx.pop()
