#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import boostmpi as mpi

MAX_GRID = 65535
nof = np.nbytes['float32']	# nbytes of float
myrank = mpi.world.rank

kernels = """
texture<float, 1, cudaReadModeElementType> tcex;
texture<float, 1, cudaReadModeElementType> tcey;
texture<float, 1, cudaReadModeElementType> tcez;
texture<float, 1, cudaReadModeElementType> tchx;
texture<float, 1, cudaReadModeElementType> tchy;
texture<float, 1, cudaReadModeElementType> tchz;

__global__ void update_e(int nx, int ny, int nz, int idx0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx + idx0;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x+1];
	float* sz = (float*) &sy[blockDim.x+1];

	sx[tx] = hx[idx];
	sy[tx] = hy[idx];
	sz[tx] = hz[idx];
	__syncthreads();

	float hy_x = sy[tx+1];
	float hz_x = sz[tx+1];
	if( tx == blockDim.x - 1 ) {
		hy_x = hy[idx+1];
		hz_x = hz[idx+1];
	}

	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	if( j<ny-1 && k<nz-1 ) ex[idx] += tex1Dfetch(tcex,idx)*( hz[idx+nx] - sz[tx] - hy[idx+nx*ny] + sy[tx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += tex1Dfetch(tcey,idx)*( hx[idx+nx*ny] - sx[tx] - hz_x + sz[tx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += tex1Dfetch(tcez,idx)*( hy_x - sy[tx] - hx[idx+nx] + sx[tx] );
}

__global__ void update_h(int nx, int ny, int nz, int idx0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx + idx0;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x];
	float* sz = (float*) &sy[blockDim.x];

	sx[tx] = ex[idx];
	sy[tx] = ey[idx];
	sz[tx] = ez[idx];
	__syncthreads();

	float hy_x = sy[tx-1];
	float hz_x = sz[tx-1];
	if( tx == 0 ) {
		hy_x = ey[idx-1];
		hz_x = ez[idx-1];
	}

	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	if( j>0 && k>0 ) hx[idx] -= tex1Dfetch(tchx,idx)*( sz[tx] - ez[idx-nx] - sy[tx] + ey[idx-nx*ny] );
	if( i>0 && k>0 ) hy[idx] -= tex1Dfetch(tchy,idx)*( sx[tx] - ex[idx-nx*ny] - sz[tx] + hz_x );
	if( i>0 && j>0 ) hz[idx] -= tex1Dfetch(tchz,idx)*( sy[tx] - hy_x - sx[tx] + ex[idx-nx] );
}

__global__ void update_src(int nx, int ny, int nz, int tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nz/2)*nx*ny + (ny/2)*nx + idx;

	if( idx < nx ) f[ijk] += sin(0.1*tn);
}

__global__ void update_e_side(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *ex_map, float *ey_map) {
	float val;
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x+1];
	float* sz = (float*) &sy[blockDim.x+1];

	sx[tx] = hx[idx];
	sy[tx] = hy[idx];
	sz[tx] = hz[idx];
	__syncthreads();

	float hy_x = sy[tx+1];
	float hz_x = sz[tx+1];
	if( tx == blockDim.x - 1 ) {
		hy_x = hy[idx+1];
		hz_x = hz[idx+1];
	}

	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	if( j<ny-1 && k<nz-1 ) {
		val = ex[idx] + tex1Dfetch(tcex,idx)*( hz[idx+nx] - sz[tx] - hy[idx+nx*ny] + sy[tx] );
		ex[idx] = val;
		ex_map[idx] = val;
	}
	if( i<nx-1 && k<nz-1 ) {
		val = ey[idx] + tex1Dfetch(tcey,idx)*( hx[idx+nx*ny] - sx[tx] - hz_x + sz[tx] );
		ey[idx] = val;
		ey_map[idx] = val;
	}
	if( i<nx-1 && j<ny-1 ) ez[idx] += tex1Dfetch(tcez,idx)*( hy_x - sy[tx] - hx[idx+nx] + sx[tx] );
}

__global__ void update_h_side(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *hx_map, float *hy_map) {
	float val;
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx + nx*ny*(nz-1);

	extern __shared__ float s[];
	float* sx = (float*) s;
	float* sy = (float*) &sx[blockDim.x];
	float* sz = (float*) &sy[blockDim.x];

	sx[tx] = ex[idx];
	sy[tx] = ey[idx];
	sz[tx] = ez[idx];
	__syncthreads();

	float hy_x = sy[tx-1];
	float hz_x = sz[tx-1];
	if( tx == 0 ) {
		hy_x = ey[idx-1];
		hz_x = ez[idx-1];
	}

	int k = idx/(nx*ny);
	int j = (idx - k*nx*ny)/nx;
	int i = idx%nx;

	if( j>0 && k>0 ) {
		val = hx[idx] - tex1Dfetch(tchx,idx)*( sz[tx] - ez[idx-nx] - sy[tx] + ey[idx-nx*ny] );
		hx[idx] = val;
		hx_map[idx] = val;
	}
	if( i>0 && k>0 ) {
		val = hy[idx] - tex1Dfetch(tchy,idx)*( sx[tx] - ex[idx-nx*ny] - sz[tx] + hz_x );
		hy[idx] = val;
		hy_map[idx] = val;
	}
	if( i>0 && j>0 ) hz[idx] -= tex1Dfetch(tchz,idx)*( sy[tx] - hy_x - sx[tx] + ex[idx-nx] );
}

"""


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

	return cf


def send(target, nx, ny, nz, fx_gpu, fy_gpu):
	if target < myrank: 
		offset_fx = int(fx_gpu)
		offset_fy = int(fy_gpu)
	else: 
		offset_fx = int(fx_gpu) + nx*ny*(nz-1)*nof
		offset_fy = int(fy_gpu) + nx*ny*(nz-1)*nof

	mpi.world.send(target, 0, cuda.from_device(offset_fx, (nx,ny), np.float32))
	mpi.world.send(target, 1, cuda.from_device(offset_fy, (nx,ny), np.float32))


def recv(target, nx, ny, nz, fx_gpu, fy_gpu):
	if target < myrank: 
		offset_fx = int(fx_gpu)
		offset_fy = int(fy_gpu)
	else: 
		offset_fx = int(fx_gpu) + nx*ny*(nz-1)*nof
		offset_fy = int(fy_gpu) + nx*ny*(nz-1)*nof

	cuda.memcpy_htod(offset_fx, mpi.world.recv(target, 0))
	cuda.memcpy_htod(offset_fy, mpi.world.recv(target, 1))
	


if __name__ == '__main__':
	cuda.init()
	ngpu = cuda.Device.count()
	#ngpu = 2
	if mpi.size != ngpu+1: 
		if myrank == 0: print 'Error: mpi.size and ngpu are not matched!'
		import sys
		sys.exit()

	nx, ny, nz = 512, 512, 335

	if myrank == 0:

		print 'dim (%d, %d, %d)' % (nx, ny, nz)
		total_bytes = nx*ny*nz*4*12
		if total_bytes/(1024**3) == 0:
			print 'mem %d MB' % ( total_bytes/(1024**2) )
		else:
			print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	else:
		ctx = cuda.Device(myrank-1).make_context(cuda.ctx_flags.MAP_HOST)

		# memory allocation
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

		# pinned memory allocation for zero-copy
		if myrank != 1:
			ex_send = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			ey_send = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			hx_recv = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			hy_recv = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			'''
			ex_send_map = ex_send.get_device_pointer()
			ey_send_map = ey_send.get_device_pointer()
			hx_recv_map = hx_recv.get_device_pointer()
			hy_recv_map = hy_recv.get_device_pointer()
			'''
		if myrank != 3:
			ex_recv = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			ey_recv = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			hx_send = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			hy_send = cuda.pagelocked_zeros((nx,ny), np.float32, order='F',  mem_flags=cuda.host_alloc_flags.DEVICEMAP)
			'''
			ex_recv_map = ex_recv.get_device_pointer()
			ey_recv_map = ey_recv.get_device_pointer()
			hx_send_map = hx_send.get_device_pointer()
			hy_send_map = hy_send.get_device_pointer()
			'''
		

		# prepare kernels
		from pycuda.compiler import SourceModule
		mod = SourceModule(kernels)
		update_e = mod.get_function("update_e")
		update_h = mod.get_function("update_h")
		update_e_side = mod.get_function("update_e_side")
		update_h_side = mod.get_function("update_h_side")
		update_src = mod.get_function("update_src")

		# bind a texture reference to linear memory
		tcex = mod.get_texref("tcex")
		tcey = mod.get_texref("tcey")
		tcez = mod.get_texref("tcez")
		tchx = mod.get_texref("tchx")
		tchy = mod.get_texref("tchy")
		tchz = mod.get_texref("tchz")

		tcex.set_address(cex_gpu, f.nbytes)
		tcey.set_address(cey_gpu, f.nbytes)
		tcez.set_address(cez_gpu, f.nbytes)
		tchx.set_address(chx_gpu, f.nbytes)
		tchy.set_address(chy_gpu, f.nbytes)
		tchz.set_address(chz_gpu, f.nbytes)


		tpb = 256
		Db = (tpb,1,1)

		bpg = (nx*ny*(nz-1))/tpb
		bpg_side = (nx*ny)/tpb

		Dg_side = (bpg_side, 1)

		Dg_list = [ (bpg%MAX_GRID, 1) ]
		idx0_e_list = [ np.int32(nx*ny) ]
		idx0_h_list = [ np.int32(0) ]
		ng = int( np.ceil( float(bpg)/MAX_GRID ) )
		for i in range(1, ng): 
			Dg_list.insert( 0, (MAX_GRID, 1) )
			idx0_e_list.append( np.int32(nx*ny + MAX_GRID*tpb*(i)) )
			idx0_h_list.append( np.int32(MAX_GRID*tpb*(i)) )

		nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)
		update_e.prepare("iiiiPPPPPP", block=Db, texrefs=[tcex,tcey,tcez], shared=tpb*3*4)
		update_h.prepare("iiiiPPPPPP", block=Db, texrefs=[tchx,tchy,tchz], shared=tpb*3*4)
		update_e_side.prepare("iiiPPPPPPPP", block=Db, texrefs=[tcex,tcey,tcez], shared=tpb*3*4)
		update_h_side.prepare("iiiPPPPPPPP", block=Db, texrefs=[tchx,tchy,tchz], shared=tpb*3*4)
		update_src.prepare("iiiiP", block=(512,1,1))

	'''
	if myrank == 0:
		# prepare for plot
		from matplotlib.pyplot import *
		ion()
		output = np.ones((ny,3*nz), 'f', order='F')
		#output = np.ones((ny,nz), 'f', order='F')
		imsh = imshow(output, cmap=cm.hot, origin='lower', vmin=0, vmax=0.001)
		colorbar()
	'''

	if myrank == 1:
		# measure kernel execution time
		#from datetime import datetime
		#t1 = datetime.now()
		start = cuda.Event()
		stop = cuda.Event()
		start.record()


	if myrank == 1:
		for i, Dg in enumerate(Dg_list): update_e.prepared_call(
				Dg, nnx, nny, nnz, idx0_e_list[i],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		#ex_recv[:,:] = mpi.world.recv(myrank+1, 0)
		#ey_recv[:,:] = mpi.world.recv(myrank+1, 1)

		cuda.memcpy_htod( int(ex_gpu) + nx*ny*(nz-1)*nof, ex_recv )
		cuda.memcpy_htod( int(ey_gpu) + nx*ny*(nz-1)*nof, ey_recv )

		update_h_side.prepared_call(
			Dg_side, nnx, nny, nnz, 
			ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, np.intp(hx_send), np.intp(hy_send) )

		for i, Dg in enumerate(Dg_list): update_h.prepared_call(
				Dg, nnx, nny, nnz, idx0_h_list[i],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		#mpi.world.send(myrank+1, 0, hx_send)
		#mpi.world.send(myrank+1, 1, hy_send)

	"""
	if myrank == 2:
		update_e_side.prepared_call(
			Dg_side, nnx, nny, nnz, 
			ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, ex_send, ey_send)

		for i, Dg in enumerate(Dg_list): update_e.prepared_call(
				Dg, nnx, nny, nnz, idx0_e_list[i],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		mpi.world.send(myrank-1, 0, ex_send)
		mpi.world.send(myrank-1, 1, ey_send)
		ex_recv[:,:] = mpi.world.recv(myrank+1, 0)
		ey_recv[:,:] = mpi.world.recv(myrank+1, 1)

		cuda.memcpy_htod( int(ex_gpu) + nx*ny*(nz-1)*nof, ex_recv )
		cuda.memcpy_htod( int(ey_gpu) + nx*ny*(nz-1)*nof, ey_recv )

		update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

		update_h_side.prepared_call(
			Dg_side, nnx, nny, nnz, 
			ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, hx_send, hy_send)

		for i, Dg in enumerate(Dg_list): update_h.prepared_call(
				Dg, nnx, nny, nnz, idx0_h_list[i],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		hx_recv[:,:] = mpi.world.recv(myrank-1, 0)
		hy_recv[:,:] = mpi.world.recv(myrank-1, 1)
		mpi.world.send(myrank+1, 0, hx_send)
		mpi.world.send(myrank+1, 1, hy_send)

		cuda.memcpy_htod( int(hx_gpu), hx_recv )
		cuda.memcpy_htod( int(hy_gpu), hy_recv )


	if myrank == 3:
		update_e_side.prepared_call(
			Dg_side, nnx, nny, nnz, 
			ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, ex_send, ey_send)

		for i, Dg in enumerate(Dg_list): update_e.prepared_call(
				Dg, nnx, nny, nnz, idx0_e_list[i],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		mpi.world.send(myrank-1, 0, ex_send)
		mpi.world.send(myrank-1, 1, ey_send)

		for i, Dg in enumerate(Dg_list): update_h.prepared_call(
				Dg, nnx, nny, nnz, idx0_h_list[i],
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

		hx_recv[:,:] = mpi.world.recv(myrank-1, 0)
		hy_recv[:,:] = mpi.world.recv(myrank-1, 1)
		
		cuda.memcpy_htod( int(hx_gpu), hx_recv )
		cuda.memcpy_htod( int(hy_gpu), hy_recv )
	"""

	"""
	# main loop
	for tn in xrange(1, 1001):
		if myrank == 1:
			for i, Dg in enumerate(Dg_list): update_e.prepared_call(
					Dg, nnx, nny, nnz, idx0_e_list[i],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

			ex_map_recv[:,:] = mpi.world.recv(myrank+1, 0)
			ey_map_recv[:,:] = mpi.world.recv(myrank+1, 1)

			update_h_side.prepared_call(
				Dg_side, nnx, nny, nnz, 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, hx_map_send, hy_map_send)

			for i, Dg in enumerate(Dg_list): update_h.prepared_call(
					Dg, nnx, nny, nnz, idx0_h_list[i],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

			mpi.world.send(myrank+1, 0, hx_map_send)
			mpi.world.send(myrank+1, 1, hy_map_send)

		if myrank == 2:
			update_e_side.prepared_call(
				Dg_side, nnx, nny, nnz, 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, ex_map_send, ey_map_send)

			for i, Dg in enumerate(Dg_list): update_e.prepared_call(
					Dg, nnx, nny, nnz, idx0_e_list[i],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

			mpi.world.send(myrank-1, 0, ex_map_send)
			mpi.world.send(myrank-1, 1, ey_map_send)
			ex_map_recv[:,:] = mpi.world.recv(myrank+1, 0)
			ey_map_recv[:,:] = mpi.world.recv(myrank+1, 1)

			update_src.prepared_call((1,1), nnx, nny, nnz, np.int32(tn), ex_gpu)

			update_h_side.prepared_call(
				Dg_side, nnx, nny, nnz, 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, hx_map_send, hy_map_send)

			for i, Dg in enumerate(Dg_list): update_h.prepared_call(
					Dg, nnx, nny, nnz, idx0_h_list[i],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

			hx_map_recv[:,:] = mpi.world.recv(myrank-1, 0)
			hy_map_recv[:,:] = mpi.world.recv(myrank-1, 1)
			mpi.world.send(myrank+1, 0, hx_map_send)
			mpi.world.send(myrank+1, 1, hy_map_send)

		if myrank == 3:
			update_e_side.prepared_call(
				Dg_side, nnx, nny, nnz, 
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, ex_map_send, ey_map_send)

			for i, Dg in enumerate(Dg_list): update_e.prepared_call(
					Dg, nnx, nny, nnz, idx0_e_list[i],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

			mpi.world.send(myrank-1, 0, ex_map_send)
			mpi.world.send(myrank-1, 1, ey_map_send)

			for i, Dg in enumerate(Dg_list): update_h.prepared_call(
					Dg, nnx, nny, nnz, idx0_h_list[i],
					ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu)

			hx_map_recv[:,:] = mpi.world.recv(myrank-1, 0)
			hy_map_recv[:,:] = mpi.world.recv(myrank-1, 1)


		'''		
		if tn%100 == 0:
			if myrank == 0:
				print 'tn =', tn
				output[:,:nz] = mpi.world.recv(1, 2)
				output[:,nz:2*nz] = mpi.world.recv(2, 2)
				output[:,2*nz:] = mpi.world.recv(3, 2)
				imsh.set_array( output[:,:]**2 )
				draw()
				#savefig('./png-wave/%.5d.png' % tstep) 
			else:
				cuda.memcpy_dtoh(f, ex_gpu)
				mpi.world.send( 0, 2, f[nx/2,:,:] )
		'''		
	"""

	if myrank == 1:
		stop.record()
		stop.synchronize()
		print stop.time_since(start)*1e-3
		#print datetime.now() - t1

	if myrank != 0: ctx.pop()
