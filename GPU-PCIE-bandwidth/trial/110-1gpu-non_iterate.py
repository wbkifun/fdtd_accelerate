#!/usr/bin/env python

import scipy as sc
import pycuda.autoinit
import pycuda.driver as cuda
from datetime import datetime


gbytes = 1024**3
mbytes = 1024**2

gbytes_float = gbytes/sc.nbytes[sc.float32]
mbytes_float = mbytes/sc.nbytes[sc.float32]


class GpuSpace:
	def __init__( s, nx ):
		s.nx = nx
		s.nbytes = nx*sc.nbytes[sc.float32]

		s.a = sc.random.rand( s.nx ).astype('f')
		s.dev_a = cuda.mem_alloc( s.nbytes )

	
	def get_transfer_rate( s ):
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

		t_rate_htod = s.nbytes/dt_htod/gbytes
		t_rate_dtoh = s.nbytes/dt_dtoh/gbytes

		return t_rate_htod, t_rate_dtoh



if __name__ == '__main__':
	obj = GpuSpace( 2*gbytes_float )
	print "Data size: %ld float ( %1.2f GB )" % ( obj.nx, obj.nbytes/gbytes )
	print "Transfer rate: %1.2f GB/s, %1.2f GB/s" % obj.get_transfer_rate()
