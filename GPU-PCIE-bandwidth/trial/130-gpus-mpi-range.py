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
	def cuda_make_context( s, i_dev ):
		s.cntxt = cuda.Device( i_dev ).make_context()


	def cuda_context_pop( s ):
		s.cntxt.pop()


	def set_nx( s, nx ):
		s.nx = nx
		s.nbytes = nx*sc.nbytes[sc.float32]
		s.a = sc.random.rand( s.nx ).astype('f')

	
	def get_transfer_rate( s ):
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

		t_rate_htod = s.nbytes/dt_htod/mbytes
		t_rate_dtoh = s.nbytes/dt_dtoh/mbytes

		return t_rate_htod, t_rate_dtoh



if __name__ == '__main__':
	m_start, m_end = 1, 1025
	m_list = range( m_start, m_end )

	if mpi.rank == 0:
		#num_gpus = cuda.Device.count()
		num_gpus = 3
		print "Number of CUDA devices: %d" % num_gpus
		print "Number of MPI size: %d" % mpi.size
		print "Data size range: 1 MB~ 1 GB "

		if num_gpus != mpi.size:
			print 'Error: mismatch the num_gpus and mpi.size'
			import sys
			sys.exit()

		t_rate_htod = sc.zeros( (num_gpus, m_end-m_start), 'f' )
		t_rate_dtoh = sc.zeros( (num_gpus, m_end-m_start), 'f' )

	obj = GpuSpace()
	obj.cuda_make_context( mpi.rank )

	t1 = datetime.now()
	for i, m in enumerate( m_list ):
		mpi.world.barrier()	
		t2 = datetime.now()
		print t2-t1, m, 'MB'
		obj.set_nx( m*mbytes_float )

		if mpi.rank == 0:
			t_rate_htod[0][i], t_rate_dtoh[0][i] = obj.get_transfer_rate()
			for dev in xrange( 1, num_gpus ):
				t_rate_htod[dev][i], t_rate_dtoh[dev][i] = mpi.world.recv( source=dev )

		else:
			mpi.world.send( dest=0, value=obj.get_transfer_rate() )
	

	if mpi.rank == 0:
		from scipy.io import write_array
		fpath = './bandwidth-mpi_barrier-%dgpus.ascii' % num_gpus
		data_list = [m_list]
		for dev in xrange( num_gpus ):
			data_list.append( t_rate_htod[dev] )
			data_list.append( t_rate_dtoh[dev] )
		write_array( fpath, sc.transpose(data_list), separator='\t', linesep='\n' )


	obj.cuda_context_pop()
