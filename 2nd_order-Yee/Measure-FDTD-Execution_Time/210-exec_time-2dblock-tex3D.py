#!/usr/bin/env python

# Constant variables (nx, ny, nz, nyz, Dx, Dy) are replaced by python string processing.
kernels ="""
__global__ void update_h(int by0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y + by0;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty+1][tx+1] = ex[idx];
	sy[ty][tx+1] = ey[idx];
	sz[ty+1][tx] = ez[idx];
	if( tx == 0 && k > 0 ) {
		sx[ty+1][0] = ex[idx-1];
		sy[ty][0] = ey[idx-1];
	}
	if( ty == 0 && j > 0 ) {
		sx[0][tx+1] = ex[idx-nz];
		sz[0][tx] = ez[idx-nz];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[ty+1][tx+1] - sx[ty+1][tx] - sz[ty+1][tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[ty][tx+1] - ey[idx-nyz] - sx[ty+1][tx+1] + sx[ty][tx+1] );
}

texture<float, 3, cudaReadModeElementType> tcex;
texture<float, 3, cudaReadModeElementType> tcey;
texture<float, 3, cudaReadModeElementType> tcez;

__global__ void update_e(int by0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y + by0;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty][tx] = hx[idx];
	sy[ty][tx] = hy[idx];
	sz[ty][tx] = hz[idx];
	if( tx == Dx-1 && k < nz-1 ) {
		sx[ty][Dx] = hx[idx+1];
		sy[ty][Dx] = hy[idx+1];
	}
	if( ty == Dy-1 && j < ny-1 ) {
		sx[Dy][tx] = hx[idx+nz];
		sz[Dy][tx] = hz[idx+nz];
	}
	__syncthreads();

	ex[idx] += tex3D(tcex,k,j,i)*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i<nx-1 ) {
		ey[idx] += tex3D(tcey,k,j,i)*( sx[ty][tx+1] - sx[ty][tx] - hz[idx+nyz] + sz[ty][tx] );
		ez[idx] += tex3D(tcez,k,j,i)*( hy[idx+nyz] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
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


def set_c(f, direction):
	f[:,:,:] = 0.5
	if 'x' in direction: f[-1,:,:] = 0
	if 'y' in direction: f[:,-1,:] = 0
	if 'z' in direction: f[:,:,-1] = 0

	return f


def arrcopy(mcopy, src, dst):
	mcopy.set_src_host( src )
	mcopy.set_dst_array( dst )
	mcopy()


nx, ny, nz = 240, 256, 256
tmax, tgap = 1000, 100

print '(%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx*ny*nz*4*9
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

Dx, Dy = 32, 16
tpb = Dx*Dy
if nz%Dx != 0:
	print "Error: nz is not multiple of %d" % (Dx)
	sys.exit()
if (nx*ny)%Dy != 0:
	print "Error: nx*ny is not multiple of %d" % (Dy)
	sys.exit()

Bx, By = nz/Dx, nx*ny/Dy	# number of block
MBy = MAX_BLOCK/Bx
bpg_list = [(Bx,MBy) for i in range(By/MBy)]
if By%MBy != 0: bpg_list.append( (Bx,By%MBy) )
#print bpg_list

# memory allocate
f = np.zeros((nx,ny,nz), 'f')
cf = np.ones_like(f)*0.5

ex_gpu = cuda.to_device(f)
ey_gpu = cuda.to_device(f)
ez_gpu = cuda.to_device(f)
hx_gpu = cuda.to_device(f)
hy_gpu = cuda.to_device(f)
hz_gpu = cuda.to_device(f)

cex = set_c(f,'yz')
cey = set_c(f,'zx')
cez = set_c(f,'xy')

descr = cuda.ArrayDescriptor3D()
descr.width = nz
descr.height = ny
descr.depth = nx
descr.format = cuda.dtype_to_array_format(f.dtype)
descr.num_channels = 1
descr.flags = 0
tcex_gpu = cuda.Array(descr)
tcey_gpu = cuda.Array(descr)
tcez_gpu = cuda.Array(descr)

mcopy = cuda.Memcpy3D()
mcopy.width_in_bytes = mcopy.src_pitch = f.strides[1]
mcopy.src_height = mcopy.height = ny
mcopy.depth = nx
arrcopy(mcopy, cex, tcex_gpu)
arrcopy(mcopy, cey, tcey_gpu)
arrcopy(mcopy, cez, tcez_gpu)

# prepare kernels
from pycuda.compiler import SourceModule
mod = SourceModule( kernels.replace('Dx',str(Dx)).replace('Dy',str(Dy)).replace('nyz',str(ny*nz)).replace('nx',str(nx)).replace('ny',str(ny)).replace('nz',str(nz)) )
update_h = mod.get_function("update_h")
update_e = mod.get_function("update_e")
update_src = mod.get_function("update_src")

tcex = mod.get_texref("tcex")
tcey = mod.get_texref("tcey")
tcez = mod.get_texref("tcez")
tcex.set_array(tcex_gpu)
tcey.set_array(tcey_gpu)
tcez.set_array(tcez_gpu)

update_h.prepare("iPPPPPP", block=(Dx,Dy,1))
update_e.prepare("iPPPPPP", block=(Dx,Dy,1), texrefs=[tcex,tcey,tcez])
update_src.prepare("fP", block=(nz,1,1))

eh_args = (ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

# prepare for plot
'''
import matplotlib.pyplot as plt
plt.ion()
imsh = plt.imshow(np.ones((nx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()
'''

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
	for i, bpg in enumerate(bpg_list): update_h.prepared_call(bpg, np.int32(i*MBy), *eh_args)
	stop2.record()
	stop2.synchronize()
	exec_time['update_h'][tn-1] = stop2.time_since(start2)
	start2.record()

	for i, bpg in enumerate(bpg_list): update_e.prepared_call(bpg, np.int32(i*MBy), *eh_args)
	stop2.record()
	stop2.synchronize()
	exec_time['update_e'][tn-1] = stop2.time_since(start2)
	start2.record()

	update_src.prepared_call((1,1), np.float32(tn), ez_gpu)
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
