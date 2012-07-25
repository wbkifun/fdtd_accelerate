#!/usr/bin/env python

import sys, os
import scipy as sc
import pycuda.driver as cuda
from datetime import datetime
import h5py

gbytes = 1024**3
mbytes = 1024**2
kbytes = 1024


usage_str = '''\
Usage:
	./bandwidth-test.py -mem_range=start,end,step,unit [ options ]
Options:
	-mem_range=start,end,step,unit
		Specify the range of data to copy
		The available unit is one of 'KB', 'MB', 'GB'
	-ngpu
		Number of GPUs used to test
		integer or 'all' is available
		default is 'all'
	-navg
		Number of execution for average
		integer
		default is 10
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
num_avg = 10
transfer_type_list = [ 'serial', 'mpi', 'mpi-barrier' ]
for argv in argv_list[1:]:
	if argv.startswith( '-mem_range=' ):
		tp = argv.replace('-mem_range=','').split(',')
		try:
			start, end, step = int(tp[0]), int(tp[1]), int(tp[2])
		except ValueError:
			print "Error: Wrong mem_range option."
			print "The start, end, step argements must be integer."
			sys.exit()
		m_list = True
		unit = tp[3]
		if unit == 'GB': num_float = gbytes/sc.nbytes[sc.float32]
		elif unit == 'MB': num_float = mbytes/sc.nbytes[sc.float32]
		elif unit == 'KB': num_float = kbytes/sc.nbytes[sc.float32]
		else:
			print "Error: Wrong mem_range option '%s'." % ( tp[3] )
			print "The available unit is one of 'GB', 'MB', 'KB'"
			sys.exit()

	elif argv.startswith( '-ngpu=' ):
		tp = argv.replace('-ngpu=','')
		if tp == 'all': pass
		else: num_gpus = int( tp )

	elif argv.startswith( '-navg=' ):
		tp = argv.replace('-navg=','')
		num_avg = int( tp )

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


print 'mem_range: (%d, %d, %d) %s' % ( start, end, step, unit )
print 'num_gpus: ', num_gpus
print 'num_avg: ', num_avg
print 'transfer_type: ', transfer_type_list
#ans = raw_input( 'Do you want to continue? (Y/n) ' )
#if ans != 'Y' and ans != 'y' and ans != '': sys.exit()


fpath = 'GPU-PCIE-bandwidth-%d_%d_%d%s-avg%d.h5' % ( start, end, step, unit, num_avg)
print fpath
fd = h5py.File( fpath )
try:
	grp_name = '%dgpu' % num_gpus
	fd.create_group( grp_name )
except ValueError:
	print "'/%s' group is already present." % grp_name
	ans = raw_input( '[R]emove or [U]se? (r/U) ' )
	if ans == 'R' or ans == 'r': 
		del fd[ grp_name ]
		fd.create_group( grp_name )
	elif ans == 'U' or ans == 'u' or ans == '': 
		pass
	else: sys.exit()
try:
	tdt_dset_name = '/%dgpu/total_dt.dset' % num_gpus
	fd.create_dataset( tdt_dset_name, (3,), 'f' )
except ValueError:
	del fd[ tdt_dset_name ]
	fd.create_dataset( tdt_dset_name, (3,), 'f' )
fd.close()

if 'serial' in transfer_type_list:
	cmd = './bandwidth-test-serial.py %d %d %d %s %d %d %d %s' % ( start, end, step, unit, num_float, num_gpus, num_avg, fpath ) 
	print '\n', '='*12, 'GPU-PCIE-bandwidth test: serial', '='*12
	print cmd
	t1 = datetime.now()
	os.system( cmd )	
	dt = datetime.now()- t1
	print dt
	fd = h5py.File( fpath )
	tdt_dset = fd[ tdt_dset_name ]
	tdt_dset[0] = dt.seconds + dt.microseconds*1e-6
	fd.close()

if 'mpi' in transfer_type_list:
	cmd = 'mpirun -np %d bandwidth-test-mpi.py %d %d %d %s %d %d %d %s %s' % ( num_gpus, start, end, step, unit, num_float, num_gpus, num_avg, fpath, 'False' ) 
	print '\n', '='*12, 'GPU-PCIE-bandwidth test: mpi', '='*12
	print cmd
	t1 = datetime.now()
	os.system( cmd )	
	dt = datetime.now()- t1
	print dt
	fd = h5py.File( fpath )
	tdt_dset = fd[ tdt_dset_name ]
	tdt_dset[1] = dt.seconds + dt.microseconds*1e-6
	fd.close()

if 'mpi-barrier' in transfer_type_list:
	cmd = 'mpirun -np %d bandwidth-test-mpi.py %d %d %d %s %d %d %d %s %s' % ( num_gpus, start, end, step, unit, num_float, num_gpus, num_avg, fpath, 'True' ) 
	print '\n', '='*12, 'GPU-PCIE-bandwidth test: mpi-barrier', '='*12
	print cmd
	t1 = datetime.now()
	os.system( cmd )	
	dt = datetime.now()- t1
	print dt
	fd = h5py.File( fpath )
	tdt_dset = fd[ tdt_dset_name ]
	tdt_dset[2] = dt.seconds + dt.microseconds*1e-6
	fd.close()
