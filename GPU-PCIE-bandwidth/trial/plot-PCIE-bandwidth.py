#!/usr/bin/env python

import scipy as sc
import h5py
import sys
from pylab import *

color_list = [ 'b', 'g', 'r', 'c', 'm', 'y' ]


Ncount = 10
fpath = sys.argv[1]
num_gpus = int( sys.argv[2].rstrip('gpu') )

fpath_list = fpath[fpath.find('_')+1:fpath.rfind('KB')].split('~')
m_start = int( fpath_list[0].rstrip('KB') )
m_end = int( fpath_list[1] )+ 1
m_list = sc.arange( m_start, m_end )
print m_start, m_end
print len(m_list)

fig = figure( dpi=150 )
ax = fig.add_subplot( 111 )

l1_list = []
l2_list = []
label_list = []
for i in range( num_gpus ):
	l1, = ax.plot( m_list, sc.ones(len(m_list)), color_list[i]+'o-' ) 
	l2, = ax.plot( m_list, sc.ones(len(m_list)), color_list[i]+'d-' ) 
	l1_list.append( l1 )
	l2_list.append( l2 )
	label_list.append( 'GPU %d'%i )
xlabel( 'Data size (KB)' )
ylabel( 'Data transfer rate (GB/s)' )
axis([1,1024,0,3])
xticks( range(0, 1025, 128) )
#legend( l1_list, label_list, loc='lower right', shadow=True )
legend( l1_list, label_list, loc='upper right', shadow=True )


fd = h5py.File( fpath )
print dict(fd)

group = fd[ '%dgpu'%(num_gpus) ]
print dict( group ).keys()

h2d = sc.zeros( (num_gpus, len(m_list)), 'f' )
d2h = sc.zeros_like( h2d )
h2d_avg = sc.zeros_like( h2d )
d2h_avg = sc.zeros_like( h2d )
for count in range( Ncount ):
	for i in range( num_gpus ):
		dset_name = 'gpu%.2d-h2d-%.3d.dset' % ( i, count )
		h2d[i,:] = group[dset_name][:]
		d2h[i,:] = group[dset_name.replace('h2d','d2h')][:]
		
		l1_list[i].set_ydata( h2d[i] )
		l2_list[i].set_ydata( d2h[i] )

		h2d_avg[i,:] += h2d[i,:]
		d2h_avg[i,:] += d2h[i,:]

	title( 'PCIE Bandwidth (%d)'%(count) )
	fpath = './fig2/%dgpu-%.3d' % ( num_gpus, count )
	savefig( fpath+ '.eps', dpi=150 )
	savefig( fpath+ '.png', dpi=150 )

	#for i in xrange( 128 ):
	#	if abs( h2d[i+1] - h2d[i] ) > 0.5:
	#		print 'point: ', i+1, 'KB'

fd.close()


# average
for i in range( num_gpus ):
	h2d_avg[i,:] /= Ncount
	d2h_avg[i,:] /= Ncount
	l1_list[i].set_ydata( h2d_avg[i] )
	l2_list[i].set_ydata( d2h_avg[i] )
title( 'PCIE Bandwidth (avg)' )
fpath = './fig/%dgpu-avg' % ( num_gpus )
savefig( fpath+ '.eps', dpi=150 )
savefig( fpath+ '.png', dpi=150 )


# fitting
from scipy import optimize
mbytes = 1024**2
fitfunc = lambda p, x: x/( 1./p[0]*x + p[1] )
errfunc = lambda p, x, y: fitfunc(p, x) - y
p0 = [5*mbytes, 0.001]
for i in range( num_gpus ):
	#p1, success = optimize.leastsq( errfunc, p0[:], args=( m_list[:64], h2d_avg[i,:64]*mbytes ) )
	p2, success = optimize.leastsq( errfunc, p0[:], args=( m_list[64:], h2d_avg[i,64:]*mbytes ) )
	p3, success = optimize.leastsq( errfunc, p0[:], args=( m_list, d2h_avg[i]*mbytes ) )
	#print p1[0]/mbytes, p1[1]
	print 'GPU', i, p2[0]/mbytes, p2[1]
	print 'GPU', i, p3[0]/mbytes, p3[1]

	l1_list[i].set_ydata( h2d_avg[i] )
	l2_list[i].set_ydata( d2h_avg[i] )
	title( 'PCIE Bandwidth (fit)' )
	#ax.plot( m_list[:64], fitfunc( p1, m_list[:64] )/mbytes, 'g-', linewidth=3 )
	ax.plot( m_list[64:], fitfunc( p2, m_list[64:] )/mbytes, 'g-', linewidth=3 )
	ax.plot( m_list, fitfunc( p3, m_list )/mbytes, 'm-', linewidth=3 )

axis([1,1024,0,3])
fpath = './fig/%dgpu-fit' % ( num_gpus )
savefig( fpath+ '.eps', dpi=150 )
savefig( fpath+ '.png', dpi=150 )
show()
