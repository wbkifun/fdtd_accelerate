#!/usr/bin/env python

import sys, os, shutil
import scipy as sc
from scipy import optimize
import h5py

fpath = sys.argv[1]
fpath2 = fpath.replace( '.h5', '-fit.h5' )
shutil.copyfile( fpath, fpath2 )

num_avg = 10

figpath_base = './figs/1gpu-x8-16KB~10MB/'
#xticks_list1 = [ 1024*2, 1024*4, 1024*6, 1024*8, 1024*10 ]
#xticks_list2 = [ 2, 4, 6, 8, 10 ]
xticks_list2 = sc.arange( 1, 11 )
xticks_list1 = xticks_list2[:]*1024
unit = 'MB'
xlim = (16, 10240)
ylim = (0, 5)
sl = slice( None, None, None )
'''
figpath_base = './figs/1gpu-x8-64KB_20MB/'
xticks_list1 = [ 1024, 1024*5, 1024*10, 1024*15, 1024*20 ]
xticks_list2 = [ 1, 5, 10, 15, 20 ]
unit = 'MB'
xlim = (64, 20480)
ylim = (0, 5)
sl = slice( None, None, None )

figpath_base = './figs/1gpu-x8-1_100MB/'
xticks_list1 = [ 1024, 1024*20, 1024*40, 1024*60, 1024*80, 1024*100 ]
xticks_list2 = [ 1, 20, 40, 60, 80, 100 ]
unit = 'MB'
xlim = (1024, 102400)
ylim = (0, 5)
sl = slice( None, None, None )

figpath_base = './figs/1gpu-x8-1_1024KB_2/'
xticks_list1 = range( 1, 1025, 128 )
xticks_list2 = range( 1, 1025, 128 )
unit = 'KB'
xlim = (1, 1024)
ylim = (0, 3)
sl = slice( 64, None, None)	#singular point
'''

args = fpath[:fpath.rfind('KB')].split('_')
start = int( args[0] )
end = int( args[1] )
step = int( args[2] )
m_list = sc.arange( start, end+1, step )
nx = len( m_list )
print 'm_list: ', m_list
print 'nx: ', nx
print 'num_avg: ', num_avg

fd = h5py.File( fpath2 )
g_ngpu = dict(fd)['1gpu']
g_ttype = dict(g_ngpu)['serial']	# transfer type

# average
dset_h2d_avg = g_ttype.create_dataset( 'gpu0-h2d-avg.dset', (nx,), 'f' )
dset_d2h_avg = g_ttype.create_dataset( 'gpu0-d2h-avg.dset', (nx,), 'f' )
for count in xrange( num_avg ):
	dset_h2d_avg[:] += g_ttype['gpu0-h2d-%d.dset'%(count)][:]
	dset_d2h_avg[:] += g_ttype['gpu0-d2h-%d.dset'%(count)][:]
dset_h2d_avg[:] /= num_avg
dset_d2h_avg[:] /= num_avg

# fitting
mbytes = 1024**2
fitfunc = lambda p, x: x/( 1./p[0]*x + p[1] )
errfunc = lambda p, x, y: fitfunc(p, x) - y
p0 = [3*mbytes, 1e-5]
p1, success = optimize.leastsq( errfunc, p0[:], args=( m_list[sl], dset_h2d_avg[sl]*mbytes ) )
p2, success = optimize.leastsq( errfunc, p0[:], args=( m_list[:], dset_d2h_avg[:]*mbytes ) )

dset_fit = g_ngpu.create_dataset( 'fit-bandwidth-latency.dset', (2, 2), 'f' )	# (h2d/d2h, bandwidth/latency)
dset_fit[0,0] = p1[0]/mbytes
dset_fit[0,1] = p1[1]/1e-6
dset_fit[1,0] = p2[0]/mbytes
dset_fit[1,1] = p2[1]/1e-6

dset_h2d_fit = g_ttype.create_dataset( 'gpu0-h2d-fit.dset', (nx,), 'f' )
dset_d2h_fit = g_ttype.create_dataset( 'gpu0-d2h-fit.dset', (nx,), 'f' )
dset_h2d_fit[sl] = fitfunc( p1, m_list[sl] )/mbytes
dset_d2h_fit[:] = fitfunc( p2, m_list[:] )/mbytes

'''
#fd2 = h5py.File( '1_1024_1KB-avg10-x8-1gpu-fit.h5' )
fd2 = h5py.File( '1024_102400_512KB-avg10-fit.h5' )
dset_fit2 = fd2['/1gpu/fit-bandwidth-latency.dset']

print (dset_fit2[0,0], dset_fit2[0,1])
print (dset_fit2[1,0], dset_fit2[1,1])

dset_h2d_fit[:] = fitfunc( (dset_fit2[0,0]*mbytes, dset_fit2[0,1]*1e-6), m_list[:] )/mbytes
dset_d2h_fit[:] = fitfunc( (dset_fit2[1,0]*mbytes, dset_fit2[1,1]*1e-6), m_list[:] )/mbytes
fd2.close()
'''

fd.close()


# plot
import matplotlib.pyplot as pl

fd = h5py.File( fpath2 )
g_ngpu = dict(fd)['1gpu']

fig = pl.figure( dpi=150 )
ax = fig.add_subplot( 111 )

#l1, = ax.plot( m_list, sc.ones(nx,'f'), 'co-' )
#l2, = ax.plot( m_list, sc.ones(nx,'f'), 'md-' )
l1, = ax.plot( m_list, sc.ones(nx,'f'), 'c.' )
l2, = ax.plot( m_list, sc.ones(nx,'f'), 'm.' )
ax.set_xlim( xlim )
ax.set_ylim( ylim )
pl.xlabel( 'Data size (%s)'%unit )
pl.ylabel( 'Data transfer rate (GB/s)' )
pl.xticks( xticks_list1, xticks_list2 )
pl.legend( (l1, l2), ('HostToDevice', 'DeviceToHost'), loc='lower right', shadow=True )

c_list = range( num_avg )
c_list.append( 'avg' )
for count in c_list:
	dset_h2d = g_ngpu['serial/gpu0-h2d-%s.dset'%str(count)]
	dset_d2h = g_ngpu['serial/gpu0-d2h-%s.dset'%str(count)]
	l1.set_ydata( dset_h2d[:] )
	l2.set_ydata( dset_d2h[:] )
	pl.title( 'PCIE Bandwidth (%s)'%str(count) )
	figpath = figpath_base + '%s' % ( str(count) )
	print figpath
	fig.savefig( figpath+ '.png', dpi=150 )

dset_h2d = g_ngpu['serial/gpu0-h2d-avg.dset']
dset_d2h = g_ngpu['serial/gpu0-d2h-avg.dset']
l1.set_ydata( dset_h2d[:] )
l2.set_ydata( dset_d2h[:] )
dset_h2d = g_ngpu['serial/gpu0-h2d-fit.dset']
dset_d2h = g_ngpu['serial/gpu0-d2h-fit.dset']
l3, = ax.plot( m_list[sl], dset_h2d[sl], 'b-', linewidth=3 )
l4, = ax.plot( m_list[:], dset_d2h[:], 'r-', linewidth=3 )
ax.set_xlim( xlim )
ax.set_ylim( ylim )
pl.title( 'PCIE Bandwidth (fit)' )
pl.legend( (l3, l4), ('HostToDevice', 'DeviceToHost'), loc='lower right', shadow=True )
figpath = figpath_base + 'fit'
print figpath
fig.savefig( figpath+ '.png', dpi=150 )

dset_fit = g_ngpu['fit-bandwidth-latency.dset']
print '\tbandwidth\tlatency'
print 'h2d:\t%g\t\t%g' % ( dset_fit[0,0], dset_fit[0,1] )
print 'd2h:\t%g\t\t%g' % ( dset_fit[1,0], dset_fit[1,1] )
