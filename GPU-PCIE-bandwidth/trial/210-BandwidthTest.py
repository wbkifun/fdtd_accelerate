#!/usr/bin/env python

import sys
import scipy as sc
import pycuda.driver as cuda
import boostmpi as mpi
from datetime import datetime

gbytes = 1024**3
mbytes = 1024**2
kbytes = 1024

cuda.init()

class BandwidthTest:
	def __init__( s, m_list ):
		s.m_list = m_list

		s.bandwidth_h2d = sc.zeros( len(m_list), 'f' )
		s.bandwidth_d2h = sc.zeros( len(m_list), 'f' )


	def cuda_make_context( s, i_dev ):
		s.cntxt = cuda.Device( i_dev ).make_context()


	def cuda_context_pop( s ):
		s.cntxt.pop()


	def allocate( s, nx ):
		s.nbytes = int( nx*sc.nbytes[sc.float32] )
		s.a = sc.random.rand( nx ).astype('f')
		s.dev_a = cuda.mem_alloc( s.nbytes )

	
	def free( s ):
		del s.nbytes, s.a, s.dev_a


	def calc_bandwidth( s, num_float, barrier=False ):
		for i, m in enumerate( s.m_list ):
			s.allocate( m*num_float )

			if barrier: mpi.world.barrier()
			t1 = datetime.now()
			cuda.memcpy_htod( s.dev_a, s.a )
			dt = datetime.now() - t1
			dt_float = dt.seconds + dt.microseconds*1e-6
			s.bandwidth_h2d[i] = s.nbytes/dt_float/gbytes

			s.free()

		for i, m in enumerate( s.m_list ):
			s.allocate( m*num_float )

			if barrier: mpi.world.barrier()
			t1 = datetime.now()
			cuda.memcpy_dtoh( s.a, s.dev_a )
			dt = datetime.now() - t1
			dt_float = dt.seconds + dt.microseconds*1e-6
			s.bandwidth_d2h[i] = s.nbytes/dt_float/gbytes

			s.free()
