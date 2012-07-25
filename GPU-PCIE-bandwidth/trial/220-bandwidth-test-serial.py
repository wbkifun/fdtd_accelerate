#!/usr/bin/env python

from BandwidthTest import *

args = sys.argv
m_list = sc.arange( int(args[1]), int(args[2])+1, int(args[3]) )
m_unit = args[4]
num_float = int( args[5] )
num_gpus = int( args[6] )
num_avg = int( args[7] )
fpath = args[8]

import h5py
fd = h5py.File( fpath )
try:
	grp_name = '/%dgpu/serial' % num_gpus
	h5group = fd.create_group( grp_name )
except ValueError:
	print "'%s' group is already present." % ( grp_name )
	ans = raw_input( 'Do you want to remove the existing group? (y/N) ' )
	if ans == 'Y' or ans == 'y': 
		del fd[ grp_name ]
		h5group = fd.create_group( grp_name )
	elif ans == 'N' or ans == 'n' or ans == '': sys.exit()


obj_list = []
for i in xrange( num_gpus ):
	obj_list.append( BandwidthTest( m_list ) )
	obj_list[i].cuda_make_context( i )

for count in range( num_avg ):
	for i, obj in enumerate( obj_list ):
		t1 = datetime.now()

		obj.calc_bandwidth( num_float )
		dset_name = 'gpu%d-h2d-%d.dset' % ( i, count )
		h5group.create_dataset( dset_name, data=obj.bandwidth_h2d[:] )
		h5group.create_dataset( dset_name.replace('h2d','d2h'), data=obj.bandwidth_d2h[:] )

		print '[%d][GPU %d] %s' % ( count, i, datetime.now()-t1 )

for obj in obj_list:
	obj.cuda_context_pop()


fd.close()
