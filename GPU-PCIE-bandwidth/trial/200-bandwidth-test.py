#!/usr/bin/env python

import sys
import scipy as sc
import pycuda.driver as cuda
import boostmpi as mpi
from datetime import datetime

gbytes = 1024**3
mbytes = 1024**2
kbytes = 1024


def get_args():
	usage_str = '''\
Usage:
	./gpu-pcie-bandwidth-test.py -mem_range=start,end,step,unit [ options ]
Options:
	-mem_range=start,end,step,unit
		Specify the range of data to copy
		The available unit is one of 'KB', 'MB', 'GB'
	-ngpu
		Number of GPUs used to test
		integer or 'all' is available
		default is 'all'
	-transfer_type
		'serial', 'mpi', 'mpi-barrier' or 'all' is available
		default is 'all'
	-h, --help
		Show this help page
Example:
	./gpu-pcie-bandwidth-test.py -mem_range=1,1024,1,KB -ngpu=3 -transfer_type=all
	'''
				
	argv_list = sys.argv
	m_list = None
	num_gpus = 'all'
	transfer_type_list = [ 'serial', 'mpi', 'mpi-barrier' ]
	for argv in argv_list[1:]:
		if argv.startswith( '-mem_range=' ):
			tp = argv.replace('-mem_range=','').split(',')
			try:
				m_list = sc.arange( int(tp[0]), int(tp[1])+1, int(tp[2]) )
			except TypeError:
				print "Error: Wrong mem_range option."
				print "The start, end, step argements must be integer."
				sys.exit()
			m_unit = tp[3]
			if m_unit == 'GB': num_float = gbytes/sc.nbytes[sc.float32]
			elif m_unit == 'MB': num_float = mbytes/sc.nbytes[sc.float32]
			elif m_unit == 'KB': num_float = kbytes/sc.nbytes[sc.float32]
			else:
				print "Error: Wrong mem_range option '%s'." % ( tp[3] )
				print "The available unit is one of 'B', 'KB', 'MB'"
				sys.exit()

		elif argv.startswith( '-ngpu=' ):
			tp = argv.replace('-ngpu=','')
			if tp == 'all': pass
			else: num_gpus = int( tp )

		elif argv.startswith( '-transfer_type=' ):
			tp = argv.replace('-transfer_type=','')
			if tp == 'all': pass
			elif tp in [ 'serial', 'mpi', 'mpi-barrier', 'all' ]:
				transfer_type_list = [ tp ]
			else:
				print "Error: Wrong transfer_type option." 
				print "The available is one of 'serial', 'mpi', 'mpi-barrier', 'all'"
				sys.exit()

		elif argv in [ '-h', '--help' ]:
			print usage_str
			sys.exit()

		else:
			print "Error: Wrong options!"
			print usage_str
			sys.exit()

	if m_list == None:
		print "Error: You must give the mem_range option!"
		print usage_str
		sys.exit()

	print 'CUDA Initialize...'
	cuda.init()
	Ngpus = cuda.Device.count()
	print 'Number of CUDA Devices: %d' % Ngpus
	if num_gpus == 'all':
		num_gpus = Ngpus
	elif num_gpus > Ngpus or num_gpus < 1:
		print "Error: Wrong ngpu option."
		print "Maximum of ngpu is %d." % ( Ngpus )
		sys.exit()

	return  m_list, m_unit, num_float, num_gpus, transfer_type_list



class BandwidthTestMulti:
	def __init__( s, m_list, num_float, num_gpus, num_avg ):
		s.m_list = m_list
		s.num_float = num_float
		s.num_gpus = num_gpus
		s.num_avg = num_avg


	def set_h5group_ngpu( s, h5group_ngpu ):
		s.h5group_ngpu = h5group_ngpu


	def serial( s ):
		try:
			grp_name = 'serial'
			h5group = s.h5group_ngpu.create_group( grp_name )
		except ValueError:
			print "'%s/%s' group is already present." % ( s.h5group_ngpu, grp_name )
			print 'Do you want to remove the existing group? (y/N)'
			ans = sys.stdin.readline()
			if ans == 'Y\n' or ans == 'y\n': 
				del s.h5group_ngpu[ grp_name ]
				h5group = s.h5group_ngpu.create_group( grp_name )
			elif ans == 'N\n' or ans == 'n\n' or ans == '\n': return None

		obj_list = []
		for i in xrange( s.num_gpus ):
			obj_list.append( BandwidthTest( i, s.m_list, s.num_float ) )
			obj_list[i].cuda_make_context()

		for count in range( s.num_avg ):
			for i in xrange( s.num_gpus ):
				t1 = datetime.now()

				obj_list[i].calc_bandwidth()
				dset_name = 'gpu%d-h2d-%d.dset' % ( i, count )
				h5group.create_dataset( dset_name, data=obj_list[i].bandwidth_h2d[:] )
				h5group.create_dataset( dset_name.replace('h2d','d2h'), data=obj_list[i].bandwidth_d2h[:] )

				print '[%d][GPU %d] %s' % ( count, i, datetime.now()-t1 )

		for obj in obj_list:
			obj.cuda_context_pop()
			del obj


	def mpi( s, barrier=False ):
		if mpi.rank == 0:
			try:
				if barrier: grp_name = 'mpi-barrier'
				else: grp_name = 'mpi'
				h5group = s.h5group_ngpu.create_group( grp_name )
			except ValueError:
				print "'%s/%s' group is already present." % ( s.h5group_ngpu, grp_name )
				print 'Do you want to remove the existing group? (y/N)'
				ans = sys.stdin.readline()
				if ans == 'Y\n' or ans == 'y\n': 
					del s.h5group_ngpu[ grp_name ]
					h5group = s.h5group_ngpu.create_group( grp_name )
				elif ans == 'N\n' or ans == 'n\n' or ans == '\n': return None

		obj = BandwidthTest( mpi.rank, s.m_list, s.num_float )
		obj.cuda_make_context()

		for count in range( num_avg ):
			t1 = datetime.now()

			obj.calc_bandwidth( barrier )
			if mpi.rank == 0:
				for dev in range( s.num_gpus ):
					dset_name = 'gpu%d-h2d-%d.dset' % ( dev, count )
					h5group.create_dataset( dset_name, data=mpi.world.recv(source=dev, tag=0) )
					h5group.create_dataset( dset_name.replace('h2d','d2h'), data=mpi.world.recv(source=dev, tag=1) )
			mpi.world.send( dest=0, tag=0, value=obj.bandwidth_h2d[:] )
			mpi.world.send( dest=0, tag=1, value=obj.bandwidth_d2h[:] )

			print '[%d][GPU %d] %s' % ( count, mpi.rank, datetime.now()-t1 )

		obj.cuda_context_pop()
		del obj



if __name__ == '__main__':
	m_list, num_float, num_gpus, num_avg, transfer_type_list = None, None, None, None, None

	if mpi.rank == 0:
		m_list, m_unit, num_float, num_gpus, transfer_type_list = get_args()
		num_avg = 10
		print 'mem_range: ', m_list, m_unit
		print 'num_gpus: ', num_gpus
		print 'transfer_type_list: ', transfer_type_list
		print 'num_avg: ', num_avg
		if num_gpus != mpi.size:
			print 'Error: mismatch the num_gpus and mpi.size'
			sys.exit()

		print 'Do you want to continue? (Y/n)' 
		ans = sys.stdin.readline()
		if ans != 'Y\n' and ans != 'y\n' and ans != '\n': sys.exit()

		total_dt = sc.zeros( 3, 'f' )

		import h5py
		fd = h5py.File( 'gpu-pcie-bandwidth-%d~%d_%d%s-avg%d.h5' % (m_list[0], m_list[-1], m_list[1]-m_list[0], m_unit, num_avg) )
		try:
			grp_name = '%dgpu' % num_gpus
			h5group_ngpu = fd.create_group( grp_name )
		except ValueError:
			print "'/%s' group is already present." % grp_name
			print 'Do you want to remove the existing group? (y/N)'
			ans = sys.stdin.readline()
			if ans == 'Y\n' or ans == 'y\n': 
				del fd[ grp_name ]
				h5group_ngpu = fd.create_group( grp_name )
			elif ans == 'N\n' or ans == 'n\n' or ans == '\n': 
				h5group_ngpu = fd[ grp_name ]
			else: sys.exit()


	m_list, num_float, num_gpus, num_avg, transfer_type_list = \
			mpi.broadcast( mpi.world, root=0, value=(m_list, num_float, num_gpus, num_avg, transfer_type_list) )
	multi_obj = BandwidthTestMulti( m_list, num_float, num_gpus, num_avg )


	if 'serial' in transfer_type_list:
		if mpi.rank == 0:
			print 'GPU-PCIE-bandwidth test: serial'
			multi_obj.set_h5group_ngpu( h5group_ngpu )
			t1 = datetime.now()
			multi_obj.serial()
			dt = datetime.now() - t1
			total_dt[0] = dt.seconds + dt.microseconds*1e-6
			print 'serial test end.', dt


	if 'mpi' in transfer_type_list:
		if mpi.rank == 0:
			print 'GPU-PCIE-bandwidth test: mpi'
			multi_obj.set_h5group_ngpu( h5group_ngpu )
			t1 = datetime.now()

		mpi.world.barrier()
		multi_obj.mpi()
		mpi.world.barrier()

		if mpi.rank == 0:
			dt = datetime.now() - t1
			total_dt[1] = dt.seconds + dt.microseconds*1e-6
			print 'mpi test end.', dt


	if 'mpi' in transfer_type_list:
		if mpi.rank == 0:
			print 'GPU-PCIE-bandwidth test: mpi-barrier'
			t1 = datetime.now()

		mpi.world.barrier()
		multi_obj.mpi( barrier=True )
		mpi.world.barrier()

		if mpi.rank == 0:
			dt = datetime.now() - t1
			total_dt[2] = dt.seconds + dt.microseconds*1e-6
			print 'mpi-barrier test end.', dt


	if mpi.rank == 0:
		fd.dataset_create( 'total_dt.dset', data=total_dt[:] )
		fd.close()
