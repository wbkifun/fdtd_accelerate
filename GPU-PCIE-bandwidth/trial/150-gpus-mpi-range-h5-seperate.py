#!/usr/bin/env python

import scipy as sc
import pycuda.driver as cuda
from datetime import datetime
import boostmpi as mpi


cuda.init()

gbytes = 1024**3
mbytes = 1024**2
kbytes = 1024
gbytes_float = gbytes/sc.nbytes[sc.float32]
mbytes_float = mbytes/sc.nbytes[sc.float32]
kbytes_float = kbytes/sc.nbytes[sc.float32]


class GpuSpace:
	def __init__( s, m_list ):
		s.m_list = m_list
		s.bandwidth_h2d = sc.zeros( len(m_list), 'f' )
		s.bandwidth_d2h = sc.zeros( len(m_list), 'f' )


	def cuda_make_context( s, i_dev ):
		s.cntxt = cuda.Device( i_dev ).make_context()


	def cuda_context_pop( s ):
		s.cntxt.pop()


	def set_nx( s, nx ):
		s.nbytes = nx*sc.nbytes[sc.float32]
		s.a = sc.random.rand( nx ).astype('f')
		s.dev_a = cuda.mem_alloc( s.nbytes )

	
	def calc_bandwidth_h2d( s ):
		t1 = datetime.now()
		cuda.memcpy_htod( s.dev_a, s.a )
		dt = datetime.now() - t1
		dt_float = dt.seconds + dt.microseconds*1e-6

		return s.nbytes/dt_float/gbytes


	def calc_bandwidth_d2h( s ):
		t1 = datetime.now()
		cuda.memcpy_dtoh( s.a, s.dev_a )
		dt = datetime.now() - t1
		dt_float = dt.seconds + dt.microseconds*1e-6

		return s.nbytes/dt_float/gbytes

	
	def calc_bandwidth_range( s, count ):
			t1 = datetime.now()
			for i, m in enumerate( s.m_list ):
				s.set_nx( m*kbytes_float )
				mpi.world.barrier()	
				s.bandwidth_h2d[i] = s.calc_bandwidth_h2d()
			print '[%d][rank %d] h2d %s' % ( count, mpi.rank, datetime.now()-t1 )

			for i, m in enumerate( s.m_list ):
				s.set_nx( m*kbytes_float )
				mpi.world.barrier()	
				s.calc_bandwidth_d2h()
				s.bandwidth_d2h[i] = s.calc_bandwidth_d2h()
			print '[%d][rank %d] d2h %s' % ( count, mpi.rank, datetime.now()-t1 )
	

if __name__ == '__main__':
	m_start, m_end = 1, 1025
	m_list = range( m_start, m_end )
	#num_gpus = cuda.Device.count()
	num_gpus = 4

	if mpi.rank == 0:
		print "Number of CUDA devices: %d" % num_gpus
		print "Number of MPI size: %d" % mpi.size
		print "Data size range: %d ~ %d KBytes " % ( m_start, m_end- 1 )

		if num_gpus != mpi.size:
			print 'Error: mismatch the num_gpus and mpi.size'
			import sys
			sys.exit()

		import h5py
		fpath = './data/PCIE-bandwidth_%dKB~%dKB-average-vridge_x100.h5' % ( m_start, m_end- 1 )
		fd = h5py.File( fpath )
		group_name = '%dgpu' % num_gpus
		fd.create_group( group_name )
		fd.close()


	obj = GpuSpace( m_list )
	obj.cuda_make_context( mpi.rank )

	for count in range( 10 ):
		obj.calc_bandwidth_range( count )

		if mpi.rank == 0:
			fd = h5py.File( fpath )
			group = fd[ group_name ]
			dset_name = 'gpu00-h2d-%.3d.dset' % ( count )
			group.create_dataset( dset_name, data=obj.bandwidth_h2d[:] )
			group.create_dataset( dset_name.replace('h2d','d2h'), data=obj.bandwidth_d2h[:] )
			for dev in xrange( 1, num_gpus ):
				dset_name2 = 'gpu%.2d%s' % ( dev, dset_name[5:] )
				group.create_dataset( dset_name2, data=mpi.world.recv(source=dev, tag=0) )
				group.create_dataset( dset_name2.replace('h2d','d2h'), data=mpi.world.recv(source=dev, tag=1) )
			fd.close()
		else:
			mpi.world.send( dest=0, tag=0, value=obj.bandwidth_h2d[:] )
			mpi.world.send( dest=0, tag=1, value=obj.bandwidth_d2h[:] )
			

	obj.cuda_context_pop()
