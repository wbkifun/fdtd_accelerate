#!/usr/bin/env python

import boostmpi as mpi
from multiprocessing import Process
import pycuda.driver as cuda
import numpy as np

cuda.init()
ngpu = cuda.Device.count()

def f(a,b):
	print mpi.rank, a, b

class Space:
	def __init__(s,nd):
		s.nd = nd

	def make_ctx(s):
		s.ctx = cuda.Device(s.nd).make_context()

	def pop_ctx(s):
		s.ctx.pop()

	def allocate(s):
		a = np.zeros((6,5),'f')
		a[:,:] = s.nd
		s.arr_gpu = cuda.to_device(a)

	def show(s):
		print cuda.from_device( s.arr_gpu, (6,5), np.float32 )

	def ff(s,a,b):
		print mpi.rank, a, b


def print_gpu(s):
	#s.show()
	print cuda.from_device( s.arr_gpu, (6,5), np.float32 )


if __name__ == '__main__':
	print mpi.rank

	p1 = Process( target=f, args=(123,'abc') )
	p1.start()

	s = Space(mpi.rank)
	Process( target=s.ff, args=(456,'def') ).start()

	s.make_ctx()
	s.allocate()
	#s.show()
	#p3 = Process( target=s.show )
	#p3.start()
	#print_gpu(s)
	Process( target=print_gpu, args=(s,) ).start()

	s.pop_ctx()


