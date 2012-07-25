#!/usr/bin/env python

import pycuda.driver as cuda
import numpy as np
#import boostmpi as mpi
from multiprocessing import Process

nof = np.nbytes['float32']	# nbytes of float

class SpaceGpu(Process):
	def __init__(s,num_device):
		Process.__init__(s)
		s.num_device = num_device

	def make_ctx(s):
		s.ctx = cuda.Device(s.num_device).make_context()

	def pop(s):
		s.ctx.pop()

	def allocate_gpu(s):
		a = np.zeros((6,5),'f')
		if s.num_device == 0:
			a[-2,:] = 1.5
		if s.num_device == 1:
			a[1,:] = 2.0
			a[-2,:] = 2.5
		if s.num_device == 2:
			a[1,:] = 3.0
		s.arr_gpu = cuda.to_device(a)
		s.nx, s.ny = s.shape = a.shape
		s.dtype = a.dtype

	def print_arr_gpu(s):
		print cuda.from_device(s.arr_gpu,s.shape,s.dtype)

	def run(s):
		print 'run', 
		s.allocate_gpu(s)
		print 'after allocate'


if __name__ == '__main__':
	cuda.init()
	ngpu = cuda.Device.count()
	print 'ngpu=', ngpu
	thread_list = []
	for i in xrange(3):
		thread = SpaceGpu(i)
		thread.make_ctx()
		thread.start()
		thread_list.append(thread)

	"""
	s = SpaceGpu()

	s.allocate_gpu(a)

	if mpi.rank == 0: print 'dev 0','\t'*5,'dev 1','\t'*5,'dev 2'
	print_arr_gpus(s)

	if mpi.rank == 0:
		s.send(1,0,'b')
		s.recv(1,1,'b')
	if mpi.rank == 1:
		s.recv(0,0,'f')
		s.send(0,1,'f')
		s.send(2,2,'b')
		s.recv(2,3,'b')
	if mpi.rank == 2:
		s.recv(1,2,'f')
		s.send(1,3,'f')

	if mpi.rank == 0: print 'After exchanging.'
	print_arr_gpus(s)
	"""
