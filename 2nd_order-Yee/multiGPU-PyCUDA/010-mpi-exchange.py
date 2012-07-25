#!/usr/bin/env python

import pycuda.driver as cuda
import numpy as np
import boostmpi as mpi

nof = np.nbytes['float32']	# nbytes of float


def send(rank, tag_mark, direction, nx, ny, a_gpu):
	if direction == 'f': offset_gpu = int(a_gpu) + ny*nof
	elif direction == 'b': offset_gpu = int(a_gpu) + (nx-2)*ny*nof
	print type(offset_gpu)
	mpi.world.send( rank, tag_mark, cuda.from_device(offset_gpu, (ny,), 'float32') )


def recv(rank, tag_mark, direction, nx, ny, a_gpu):
	if direction == 'f': offset_gpu = int(a_gpu)
	elif direction == 'b': offset_gpu = int(a_gpu) + (nx-1)*ny*nof
	cuda.memcpy_htod( offset_gpu, mpi.world.recv(rank, tag_mark) )
	

def send_result(nx, ny, a_gpu):
	mpi.world.send( 0, 10, cuda.from_device(a_gpu, (nx,ny), 'float32') )


def print_arr_gpus(ngpu, nx, ny, a_gpu):
	send_result(nx, ny, a_gpu)
	if mpi.rank == 0: 
		result = cuda.from_device(a_gpu, (nx,ny), 'float32')
		print ngpu
		for i in range(1,ngpu): 
			result = np.concatenate((result, mpi.world.recv(i,10)))
		for i in xrange(ny):
			print result[:nx,i],'\t',result[nx:2*nx,i],'\t',result[2*nx:,i]


if __name__ == '__main__':
	cuda.init()
	ngpu = cuda.Device.count()
	ctx = cuda.Device(mpi.rank).make_context()
	
	nx, ny = 6, 5

	a = np.zeros((nx,ny),'f')
	if mpi.rank == 0: 
		a[-2,:] = 1.5
	elif mpi.rank == 1: 
		a[1,:] = 2.0
		a[-2,:] = 2.5
	elif mpi.rank == 2: 
		a[1,:] = 3.0
	a_gpu = cuda.to_device(a)

	if mpi.rank == 0: print 'dev 0','\t'*5,'dev 1','\t'*5,'dev 2'
	print_arr_gpus(ngpu, nx, ny, a_gpu)

	if mpi.rank == 0:
		send(1, 0, 'b', nx, ny, a_gpu)
		recv(1, 1, 'b', nx, ny, a_gpu)
	if mpi.rank == 1:
		recv(0, 0, 'f', nx, ny, a_gpu)
		send(0, 1, 'f', nx, ny, a_gpu)
		send(2, 2, 'b', nx, ny, a_gpu)
		recv(2, 3, 'b', nx, ny, a_gpu)
	if mpi.rank == 2:
		recv(1, 2, 'f', nx, ny, a_gpu)
		send(1, 3, 'f', nx, ny, a_gpu)

	if mpi.rank == 0: print 'After exchanging.'
	print_arr_gpus(ngpu, nx, ny, a_gpu)

	ctx.pop()
