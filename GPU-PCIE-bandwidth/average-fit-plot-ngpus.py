#!/usr/bin/env python

import sys, os, shutil
import scipy as sc
from scipy import optimize
import h5py

transfer_type = ['serial', 'mpi', 'mpi-barrier']

fpath = sys.argv[1]
fpath2 = fpath.replace( '.h5', '-fit.h5' )
print 'Copy %s' % fpath2
if os.path.isfile( fpath2 ):
	ans = raw_input( 'Overwrite? (y/N) ')
	if ans == 'N' or ans == 'n' or ans == '': sys.exit()
	elif ans == 'Y' or ans == 'y': pass
	else: sys.exit()
shutil.copyfile( fpath, fpath2 )

ans = raw_input( 'Is there a singular point? (Y/n) ' )
if ans == 'Y' or ans == 'y' or ans == '': 
	singular = True
	ans2 = raw_input( 'Where is the singular point? ( (int)index, default=64 ) ')
	if ans2 == '': si = 64
	else: si = int( ans2 )
elif ans == 'N' or ans == 'n': singular = False
else: sys.exit()

args = fpath[fpath.find('bandwidth-')+10:fpath.rfind('KB')].split('_')
start = int( args[0] )
end = int( args[1] )
step = int( args[2] )
m_list = sc.arange( start, end+1, step )
nx = len( m_list )
num_avg = int( fpath[fpath.find('avg')+3:fpath.rfind('.h5')] )
print 'm_list: ', m_list
print 'nx: ', nx
print 'num_avg: ', num_avg

fd = h5py.File( fpath2 )

for key in dict( fd ):
	g_ngpu = dict(fd)[key]
	num_gpus = int( g_ngpu.name[1:].replace('gpu','') )
	dset_fit = g_ngpu.create_dataset( 'fit-bandwidth-latency.dset', (3, num_gpus, 2, 2), 'f' )	# (serial/mpi/mpi-barrier, num_gpus, h2d/d2h, bandwidth/latency)

	for key2 in dict( g_ngpu ):
		g_ttype = dict(g_ngpu)[key2]	# transfer type

		if not g_ttype.name.endswith( '.dset' ):
			for i in xrange( num_gpus ):
				# average
				dset_h2d_avg = g_ttype.create_dataset( 'gpu%d-h2d-avg.dset'%i, (nx,), 'f' )
				dset_d2h_avg = g_ttype.create_dataset( 'gpu%d-d2h-avg.dset'%i, (nx,), 'f' )
				for count in xrange( num_avg ):
					dset_h2d_avg[:] += g_ttype['gpu%d-h2d-%d.dset'%(i, count)][:]
					dset_d2h_avg[:] += g_ttype['gpu%d-d2h-%d.dset'%(i, count)][:]
				dset_h2d_avg[:] /= num_avg
				dset_d2h_avg[:] /= num_avg

				# fitting
				mbytes = 1024**2
				fitfunc = lambda p, x: x/( 1./p[0]*x + p[1] )
				errfunc = lambda p, x, y: fitfunc(p, x) - y
				p0 = [3*mbytes, 1e-5]
				if singular:
					p1, success = optimize.leastsq( errfunc, p0[:], args=( m_list[si:], dset_h2d_avg[si:]*mbytes ) )
				else:
					p1, success = optimize.leastsq( errfunc, p0[:], args=( m_list[:], dset_h2d_avg[:]*mbytes ) )
				p2, success = optimize.leastsq( errfunc, p0[:], args=( m_list[:], dset_d2h_avg[:]*mbytes ) )

				if g_ttype.name.endswith('serial'): i_ttype = 0
				elif g_ttype.name.endswith('mpi'): i_ttype = 1
				elif g_ttype.name.endswith('mpi-barrier'): i_ttype = 2
				dset_fit[i_ttype,i,0,0] = p1[0]/mbytes
				dset_fit[i_ttype,i,0,1] = p1[1]/1e-6
				dset_fit[i_ttype,i,1,0] = p2[0]/mbytes
				dset_fit[i_ttype,i,1,1] = p2[1]/1e-6


num_ngpus = len( fd.keys() )
dset_fit_serial = fd.create_dataset( 'fit-bandwidth-latency-serial.dset', (num_ngpus,2,2,2), 'f' )
dset_fit_mpi = fd.create_dataset( 'fit-bandwidth-latency-mpi.dset', (num_ngpus,2,2,2), 'f' )
dset_fit_mpi_barrier = fd.create_dataset( 'fit-bandwidth-latency-mpi-barrier.dset', (num_ngpus,2,2,2), 'f' )
for i in xrange( num_ngpus ):
	g_ngpu = fd['%dgpu'%(i+1)]
	dset_fit = g_ngpu['fit-bandwidth-latency.dset']
	for j in xrange(2):			# h2d/d2h
		for k in xrange(2):		# bandwidth/latency
			dset_fit_serial[i,j,k,0] = dset_fit[0,:,j,k].mean()
			dset_fit_mpi[i,j,k,0] = dset_fit[1,:,j,k].mean()
			dset_fit_mpi_barrier[i,j,k,0] = dset_fit[2,:,j,k].mean()

			dset_fit_serial[i,j,k,1] = dset_fit[0,:,j,k].std()
			dset_fit_mpi[i,j,k,1] = dset_fit[1,:,j,k].std()
			dset_fit_mpi_barrier[i,j,k,1] = dset_fit[2,:,j,k].std()

fd.close()
