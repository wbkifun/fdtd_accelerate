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

	
	def get_bandwidth( s ):
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

		bandwidth_htod = s.nbytes/dt_htod/gbytes
		bandwidth_dtoh = s.nbytes/dt_dtoh/gbytes

		return bandwidth_htod, bandwidth_dtoh



if __name__ == '__main__':
	m_start, m_end = 1, 1025
	m_list = range( m_start, m_end )

	if mpi.rank == 0:
		#num_gpus = cuda.Device.count()
		num_gpus = 2
		print "Number of CUDA devices: %d" % num_gpus
		print "Number of MPI size: %d" % mpi.size
		print "Data size range: %d ~ %d MBytes " % ( m_start, m_end- 1 )

		if num_gpus != mpi.size:
			print 'Error: mismatch the num_gpus and mpi.size'
			import sys
			sys.exit()

		bandwidth_htod = sc.zeros( (num_gpus, m_end-m_start), 'f' )
		bandwidth_dtoh = sc.zeros( (num_gpus, m_end-m_start), 'f' )

	obj = GpuSpace()
	obj.cuda_make_context( mpi.rank )

	t1 = datetime.now()
	for i, m in enumerate( m_list ):
		mpi.world.barrier()	
		t2 = datetime.now()
		print t2-t1, m, 'MB'
		obj.set_nx( m*mbytes_float )

		if mpi.rank == 0:
			bandwidth_htod[0][i], bandwidth_dtoh[0][i] = obj.get_bandwidth()
			for dev in xrange( 1, num_gpus ):
				bandwidth_htod[dev][i], bandwidth_dtoh[dev][i] = mpi.world.recv( source=dev )

		else:
			mpi.world.send( dest=0, value=obj.get_bandwidth() )
	

	if mpi.rank == 0:
		import h5py
		fpath = './benchmark-PCIE-bandwidth_%dMB~%dMB.h5' % ( m_start, m_end- 1 )
		fd = h5py.File( fpath )
		group_name = '%dgpu' % num_gpus
		group = fd.create_group( group_name )
		group.create_dataset( 'htod.dset', data=bandwidth_htod )
		group.create_dataset( 'dtoh.dset', data=bandwidth_dtoh )
		fd.close()

	obj.cuda_context_pop()
