#!/usr/bin/env python

import scipy as sc
import pycuda.driver as cuda
from datetime import datetime
import boostmpi as mpi


cuda.init()

gbytes = 1024**3
mbytes = 1024**2
gbytes_float = gbytes/sc.nbytes[sc.float32]
mbytes_float = mbytes/sc.nbytes[sc.float32]


class GpuSpace:
	def __init__( s, i_dev, nx ):
		s.nx = nx
		s.nbytes = nx*sc.nbytes[sc.float32]
		s.i_dev = i_dev

		s.a = sc.random.rand( s.nx ).astype('f')

	
	def get_transfer_rate( s ):
		cntxt = cuda.Device( s.i_dev ).make_context()

		s.dev_a = cuda.mem_alloc( s.nbytes )

		t1 = datetime.now()
		cuda.memcpy_htod( s.dev_a, s.a )
		t2 = datetime.now()
		dt = t2 - t1
		dt_htod = dt.seconds + dt.microseconds*1e-6

		t1 = datetime.now()
		cuda.memcpy_dtoh( s.a, s.dev_a )
		t2 = datetime.now()
		dt = t2 - t1
		dt_dtoh = dt.seconds + dt.microseconds*1e-6

		cntxt.pop()

		t_rate_htod = s.nbytes/dt_htod/gbytes
		t_rate_dtoh = s.nbytes/dt_dtoh/gbytes

		return s.i_dev, t_rate_htod, t_rate_dtoh



if __name__ == '__main__':
	nx = 2000*mbytes_float

	obj = GpuSpace( mpi.rank, nx )

	if mpi.rank == 0:
		num_gpus = cuda.Device.count()
		print "Number of CUDA devices: %d" % num_gpus
		print "Data size: %ld float ( %1.2f MB )" % ( obj.nx, obj.nbytes/mbytes )

		print "Transfer rate:"

	print "[GPU %d] %1.2f GB/s, %1.2f GB/s" % obj.get_transfer_rate()
