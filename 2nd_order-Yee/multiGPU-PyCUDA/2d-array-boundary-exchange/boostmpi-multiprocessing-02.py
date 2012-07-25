#!/usr/bin/env python

import boostmpi as mpi
from multiprocessing import Process
import pycuda.driver as cuda
import numpy as np


class SpaceGpu:
	def allocate(s,val):
		a = np.zeros((6,5),'f')
		a[:,:] = val
		s.arr_gpu = cuda.to_device(a)

	def send(s,rank,tag_mark):
		a = cuda.from_device(s.arr_gpu,(6,5),np.float32)
		mpi.world.send( rank,tag_mark, a)

	def recv(s,rank,tag_mark):
		cuda.memcpy_htod( s.arr_gpu, mpi.world.recv(rank,tag_mark) )

	def show(s, a):
		print a
		print cuda.from_device( s.arr_gpu, (6,5), np.float32 ).T


if __name__ == '__main__':
	cuda.init()
	ngpu = cuda.Device.count()
	ctx = cuda.Device(mpi.rank).make_context()
	s = SpaceGpu()

	s.allocate(mpi.rank)

	s.show(123)
	
	if mpi.rank == 0:
		#s.send(1,0)
		Process( target=s.send, args=(1,0) ).start()
	if mpi.rank == 1:
		s.recv(0,0)
		#Process( target=s.recv, args=(0,0) ).start()

	#s.show()
	Process( target=s.show, args=(456,) ).start()
	
	ctx.pop()


