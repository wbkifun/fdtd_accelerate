#!/usr/bin/env python

from BandwidthTest import *

args = sys.argv
m_list = sc.arange( int(args[1]), int(args[2])+1, int(args[3]) )
m_unit = args[4]
num_float = int( args[5] )
num_gpus = int( args[6] )
num_avg = int( args[7] )
fpath = args[8]
barrier = bool( args[9]=='True' )

if mpi.rank == 0:
	import h5py
	fd = h5py.File( fpath )
	try:
		if not barrier: grp_name = '/%dgpu/mpi' % num_gpus
		else: grp_name = '/%dgpu/mpi-barrier' % num_gpus
		h5group = fd.create_group( grp_name )
	except ValueError:
		print "'%s' group is already present." % ( grp_name )
		print 'Do you want to remove the existing group? (y/N) '
		ans = sys.stdin.readline()
		if ans == 'Y' or ans == 'y': 
			del fd[ grp_name ]
			h5group = fd.create_group( grp_name )
		elif ans == 'N' or ans == 'n' or ans == '': sys.exit()


obj = BandwidthTest( m_list )
obj.cuda_make_context( mpi.rank )

for count in range( num_avg ):
	t1 = datetime.now()

	obj.calc_bandwidth( num_float, barrier )
	mpi.world.send( dest=0, tag=0, value=obj.bandwidth_h2d[:] )
	mpi.world.send( dest=0, tag=1, value=obj.bandwidth_d2h[:] )

	if mpi.rank == 0:
		for dev in range( num_gpus ):
			dset_name = 'gpu%d-h2d-%d.dset' % ( dev, count )
			h5group.create_dataset( dset_name, data=mpi.world.recv(source=dev, tag=0) )
			h5group.create_dataset( dset_name.replace('h2d','d2h'), data=mpi.world.recv(source=dev, tag=1) )

	print '[%d][GPU %d] %s' % ( count, mpi.rank, datetime.now()-t1 )

obj.cuda_context_pop()


if mpi.rank == 0: fd.close()
