#!/usr/bin/env python

import pycuda.driver as cuda
import numpy as np
from mpi4py import MPI
from multiprocessing import Process

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nof = np.nbytes['float32']	# nbytes of float

class SpaceGpu:
	def allocate_gpu(s,arr):
		s.arr_gpu = cuda.to_device(arr)
		s.nx, s.ny = s.shape = arr.shape
		s.dtype = arr.dtype

	def send(s,rank,tag_mark,direction):
		if direction == 'f': offset_gpu = int(s.arr_gpu)+s.ny*nof
		elif direction == 'b': offset_gpu = int(s.arr_gpu)+(s.nx-2)*s.ny*nof
		print type(offset_gpu)
		comm.send( rank, tag_mark, cuda.from_device(offset_gpu, (s.ny,), s.dtype) )

	def recv(s,rank,tag_mark,direction):
		if direction == 'f': offset_gpu = int(s.arr_gpu)
		elif direction == 'b': offset_gpu = int(s.arr_gpu)+(s.nx-1)*s.ny*nof
		cuda.memcpy_htod( offset_gpu, comm.recv(rank,tag_mark) )
	
	def send_result(s):
		comm.send( 0, 10, cuda.from_device(s.arr_gpu,s.shape,s.dtype) )


def print_arr_gpus(s):
	s.send_result()
	if rank == 0: 
		result = cuda.from_device(s.arr_gpu,s.shape,s.dtype)
		for i in range(1,ngpu): 
			result = np.concatenate((result,comm.recv(i,10)))
		for i in xrange(s.ny):
			print result[:s.nx,i],'\t',result[s.nx:2*s.nx,i],'\t',result[2*s.nx:,i]


if __name__ == '__main__':
	cuda.init()
	ngpu = cuda.Device.count()
	ctx = cuda.Device(rank).make_context()
	s = SpaceGpu()

	a = np.zeros((6,5),'f')
	if rank == 0: 
		a[-2,:] = 1.5
	elif rank == 1: 
		a[1,:] = 2.0
		a[-2,:] = 2.5
	elif rank == 2: 
		a[1,:] = 3.0
	s.allocate_gpu(a)

	if rank == 0: print 'dev 0','\t'*5,'dev 1','\t'*5,'dev 2'
	print_arr_gpus(s)

	if rank == 0:
		s.send(1,0,'b')
		s.recv(1,1,'b')
	if rank == 1:
		s.recv(0,0,'f')
		s.send(0,1,'f')
		s.send(2,2,'b')
		s.recv(2,3,'b')
	if rank == 2:
		s.recv(1,2,'f')
		s.send(1,3,'f')

	if rank == 0: print 'After exchanging.'
	print_arr_gpus(s)

	ctx.pop()
