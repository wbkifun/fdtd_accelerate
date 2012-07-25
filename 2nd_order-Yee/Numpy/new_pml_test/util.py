#!/usr/bin/env python

from datetime import datetime
import sys


def print_bytes(nbytes):
	if nbytes > 1024**3:
		print ' %1.2f GiB' % ( float(nbytes)/(1024**3) )
	elif nbytes > 1024**2:
		print ' %1.2f MiB' % ( float(nbytes)/(1024**2) )
	elif nbytes > 1024:
		print ' %1.2f KiB' % ( float(nbytes)/1024 )


def print_mem(n, byte_size, arrays_points):
	nx, ny, nz = n
	points = nx*ny*nz

	print 'n: (%d, %d, %d)' % n

	print 'amount of point size: ', 
	if points > 1000**3:
		print ' %1.2f Gpoints' % ( float(points)/(1000**3) )
	elif points > 1000**2:
		print ' %1.2f Mpoints' % ( float(points)/(1000**2) )
	elif points > 1000:
		print ' %1.2f Kpoints' % ( float(points)/1000 )

	print 'amount of memory usage: '
	total_nbytes = 0
	for key in arrays_points.keys():
		print '\t%s: ' % key,
		nbytes = arrays_points[key] * byte_size
		print_bytes(nbytes)
		total_nbytes += nbytes
	print '\tTotal: ',
	print_bytes(total_nbytes)
	print ''


time_dict = {'point':None, 'flop':None, 'tmax':None, 'tgap':None, 't0':datetime.now(), 't1':datetime.now(), 'sums':[0.0, 0.0]}

def print_time(tstep):
	t2 = datetime.now()
	dt = (t2 - time_dict['t1']).seconds + (t2 - time_dict['t1']).microseconds*1e-6
	points = time_dict['point']/dt
	flops = time_dict['flop']/dt
	count = tstep/time_dict['tgap'] - 3
	if count > 0:
		time_dict['sums'][0] += points
		time_dict['sums'][1] += flops

		print "[%s] %d/%d(%d %%)  %1.2f(%1.2f) Mpoint/s  %1.2f(%1.2f) GFLOP/s\r" \
				% (t2-time_dict['t0'], tstep, \
				time_dict['tmax'], float(tstep)/time_dict['tmax']*100, \
				points*1e-6, time_dict['sums'][0]*1e-6/count, \
				flops*1e-9, time_dict['sums'][1]*1e-9/count, ),
		sys.stdout.flush()

	time_dict['t1'] = datetime.now()


