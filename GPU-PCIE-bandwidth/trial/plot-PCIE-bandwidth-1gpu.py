#!/usr/bin/env python

import scipy as sc
import h5py
import sys
from pylab import *

color_list = [ 'b', 'g', 'r', 'c', 'm', 'y' ]


Ncount = 10
fpath = sys.argv[1]

fpath_list = fpath[fpath.rfind('_')+1:fpath.rfind('KB')].split('~')
m_start = int( fpath_list[0].rstrip('KB') )
m_end = int( fpath_list[1] )+ 1
m_list = sc.arange( m_start, m_end )
print m_start, m_end
print len(m_list)

fig = figure( dpi=150 )
ax = fig.add_subplot( 111 )

l1, = ax.plot( m_list, sc.ones(len(m_list)), 'bo-' )
l2, = ax.plot( m_list, sc.ones(len(m_list)), 'rd-' )
xlabel( 'Data size (KB)' )
ylabel( 'Data transfer rate (GB/s)' )
axis([1,1024,0,3])
xticks( range(0, 1025, 128) )
legend( (l1, l2), ('Host to Device', 'Device to Host' ), loc='lower right', shadow=True )


fd = h5py.File( fpath )
print dict(fd)

group = fd[ '1gpu' ]
print dict( group ).keys()

h2d = sc.zeros( len(m_list), 'f' )
d2h = sc.zeros_like( h2d )
h2d_avg = sc.zeros_like( h2d )
d2h_avg = sc.zeros_like( h2d )
for count in range( Ncount ):
	dset_name = 'gpu00-h2d-%.3d.dset' % ( count )
	h2d[:] = group[dset_name][:]
	d2h[:] = group[dset_name.replace('h2d','d2h')][:]
	
	l1.set_ydata( h2d )
	l2.set_ydata( d2h )
	title( 'PCIE Bandwidth (%d)'%(count) )
	fpath = './fig/1gpu-%.3d' % count
	savefig( fpath+ '.eps', dpi=150 )
	savefig( fpath+ '.png', dpi=150 )

	for i in xrange( 128 ):
		if abs( h2d[i+1] - h2d[i] ) > 0.5:
			print 'point: ', i+1, 'KB'
	h2d_avg[:] += h2d[:]
	d2h_avg[:] += d2h[:]

fd.close()

# average
h2d_avg[:] /= Ncount
d2h_avg[:] /= Ncount
l1.set_ydata( h2d_avg )
l2.set_ydata( d2h_avg )
title( 'PCIE Bandwidth (avg)' )
fpath = './fig/1gpu-avg'
savefig( fpath+ '.eps', dpi=150 )
savefig( fpath+ '.png', dpi=150 )

# fitting
from scipy import optimize
mbytes = 1024**2
fitfunc = lambda p, x: x/( 1./p[0]*x + p[1] )
errfunc = lambda p, x, y: fitfunc(p, x) - y
p0 = [5*mbytes, 0.001]
#p1, success = optimize.leastsq( errfunc, p0[:], args=( m_list[:64], h2d_avg[:64]*mbytes ) )
p2, success = optimize.leastsq( errfunc, p0[:], args=( m_list[64:], h2d_avg[64:]*mbytes ) )
p3, success = optimize.leastsq( errfunc, p0[:], args=( m_list, d2h_avg*mbytes ) )
#print p1[0]/mbytes, p1[1]
print p2[0]/mbytes, p2[1]
print p3[0]/mbytes, p3[1]

l1.set_ydata( h2d_avg )
l2.set_ydata( d2h_avg )
title( 'PCIE Bandwidth (fit)' )
#ax.plot( m_list[:64], fitfunc( p1, m_list[:64] )/mbytes, 'g-', linewidth=3 )
ax.plot( m_list[64:], fitfunc( p2, m_list[64:] )/mbytes, 'g-', linewidth=3 )
ax.plot( m_list, fitfunc( p3, m_list )/mbytes, 'm-', linewidth=3 )

fpath = './fig/1gpu-fit'
savefig( fpath+ '.eps', dpi=150 )
savefig( fpath+ '.png', dpi=150 )
show()
