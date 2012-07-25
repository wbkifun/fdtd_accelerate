#!/usr/bin/env python

import scipy as sc
import pycuda.driver
import datetime
import pp

pycuda.driver.init()

gbytes = 1024**3
mbytes = 1024**2
gbytes_float = gbytes/sc.nbytes[sc.float32]
mbytes_float = mbytes/sc.nbytes[sc.float32]


class GpuSpace:
	def __init__( s, i_dev, nx ):
		s.nx = nx
		s.nbytes = nx*sc.nbytes[sc.float32]
		s.i_dev = i_dev

		#s.a = sc.random.rand( s.nx ).astype('f')
		s.a = sc.ones( s.nx, 'f' )

	
	def get_transfer_rate( s ):
		pycuda.driver.init()
		cntxt = pycuda.driver.Device( s.i_dev ).make_context()

		s.dev_a = pycuda.driver.mem_alloc( s.nbytes )

		t1 = datetime.datetime.now()
		pycuda.driver.memcpy_htod( s.dev_a, s.a )
		t2 = datetime.datetime.now()
		dt = t2 - t1
		dt_htod = dt.seconds + dt.microseconds*1e-6

		t1 = datetime.datetime.now()
		pycuda.driver.memcpy_dtoh( s.a, s.dev_a )
		t2 = datetime.datetime.now()
		dt = t2 - t1
		dt_dtoh = dt.seconds + dt.microseconds*1e-6

		cntxt.pop()

		t_rate_htod = s.nbytes/dt_htod/1024**3
		t_rate_dtoh = s.nbytes/dt_dtoh/1024**3

		return t_rate_htod, t_rate_dtoh



if __name__ == '__main__':
	nx = 1000*mbytes_float

	num_gpus = pycuda.driver.Device.count()
	print "Number of CUDA devices: %d" % num_gpus
	obj0 = GpuSpace( 0, nx )
	obj1 = GpuSpace( 1, nx )

	t1 = datetime.datetime.now()
	job_server = pp.Server( ncpus=num_gpus )
	t2 = datetime.datetime.now()
	print t2 - t1
	t1 = datetime.datetime.now()
	f0 = job_server.submit( obj0.get_transfer_rate, modules=("pycuda.driver","datetime"), globals=globals() )
	t2 = datetime.datetime.now()
	print t2 - t1
	t1 = datetime.datetime.now()
	f1 = job_server.submit( obj1.get_transfer_rate, modules=("pycuda.driver","datetime"), globals=globals() )
	t2 = datetime.datetime.now()
	print t2 - t1

	print "Data size: %ld float ( %1.2f MB )" % ( obj0.nx, obj0.nbytes/mbytes )
	print "Transfer rate:"
	print "[GPU 0] %1.2f GB/s, %1.2f GB/s" % f0()
	print "[GPU 1] %1.2f GB/s, %1.2f GB/s" % f1()
