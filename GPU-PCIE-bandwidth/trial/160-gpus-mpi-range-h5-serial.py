#!/usr/bin/env python

import scipy as sc
import pycuda.driver as cuda
from datetime import datetime


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

	
	def calc_bandwidth_range( s, count, dev ):
		t1 = datetime.now()
		for i, m in enumerate( s.m_list ):
			s.set_nx( m*kbytes_float )
			s.bandwidth_h2d[i] = s.calc_bandwidth_h2d()
		print '[%d][GPU %d] h2d %s' % ( count, dev, datetime.now()-t1 )

		for i, m in enumerate( s.m_list ):
			s.set_nx( m*kbytes_float )
			s.calc_bandwidth_d2h()
			s.bandwidth_d2h[i] = s.calc_bandwidth_d2h()
		print '[%d][GPU %d] d2h %s' % ( count, dev, datetime.now()-t1 )
	

if __name__ == '__main__':
	m_start, m_end = 1, 1025
	m_list = range( m_start, m_end )
	#num_gpus = cuda.Device.count()
	num_gpus = 4

	print "Number of CUDA devices: %d" % num_gpus
	print "Data size range: %d ~ %d KBytes " % ( m_start, m_end- 1 )

	import h5py
	fpath = './data/PCIE-bandwidth_%dKB~%dKB-average-serial-vridge_x100.h5' % ( m_start, m_end- 1 )
	fd = h5py.File( fpath )
	group_name = '%dgpu' % num_gpus
	group = fd.create_group( group_name )

	obj_list = []
	for i in xrange( num_gpus ):
		obj_list.append( GpuSpace( m_list ) )
		obj_list[i].cuda_make_context( i )

	for count in range( 10 ):
		for i in xrange( num_gpus ):
			obj_list[i].calc_bandwidth_range( count, i )
			dset_name = 'gpu%.2d-h2d-%.3d.dset' % ( i, count )
			group.create_dataset( dset_name, data=obj_list[i].bandwidth_h2d[:] )
			group.create_dataset( dset_name.replace('h2d','d2h'), data=obj_list[i].bandwidth_d2h[:] )

	fd.close()
	for i in xrange( num_gpus ):
		obj_list[i].cuda_context_pop()
