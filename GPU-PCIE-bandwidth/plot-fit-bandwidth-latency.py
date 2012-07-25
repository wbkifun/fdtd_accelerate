#!/usr/bin/env python

import sys
import scipy as sc
import h5py
import matplotlib.pyplot as pl


class PlotFitBL:
	def __init__( s, num_ngpus ):
		s.num_ngpus = num_ngpus

	
	def autolabel( s, ax, bars ):
		# attach some text labels
		for bar in bars:
			height = bar.get_height()
			ax.text(bar.get_x()+bar.get_width()/3., 1.05*height, '%d'%int(height), ha='center', va='bottom')

									
	def savefig( s, dset, ttype, figpath ):
		fig = pl.figure( dpi=150 )
		ax1 = fig.add_subplot( 111 )
		ax2 = ax1.twinx()

		width = 0.1
		gap = 0.05
		dx = width + gap
		x0 = sc.arange( num_ngpus ) 
		x = x0 + (width*4 + gap*3)/2 - gap
		b1 = ax1.bar( x, dset[:,0,0,0], width, color='c', yerr=dset[:,0,0,1] )
		b2 = ax1.bar( x+dx, dset[:,1,0,0], width, color='b', yerr=dset[:,1,0,1] )
		b3 = ax2.bar( x+2*dx, dset[:,0,1,0], width, color='m', yerr=dset[:,0,1,1] )
		b4 = ax2.bar( x+3*dx, dset[:,1,1,0], width, color='r', yerr=dset[:,1,1,1] )

		ax1.set_ylim( 0, 3.5 )
		ax2.set_ylim( 0, 40 )
		ax1.set_title( 'GPU-PCIE Bandwidth, Latency (%s)' % ttype )
		ax1.set_ylabel('Bandwidth (GB/s)')
		ax2.set_ylabel('Latency (us)')
		ax1.set_xlabel('Number of GPUs')
		ax1.set_xticks( x0 + 0.5 )
		ax1.set_xticklabels( x0 + 1 )
		ax2.legend( (b1[0], b2[0], b3[0], b4[0]), ('bandwidth h2d', 'bandwidth d2h', 'latency h2d', 'latency d2h'), loc='lower left' )
		s.autolabel( ax1, b1 )
		s.autolabel( ax1, b2 )
		s.autolabel( ax2, b3 )
		s.autolabel( ax2, b4 )

		fig.savefig( figpath, dpi=150 )
		fig.clf()



if __name__ == '__main__':
	fpath = sys.argv[1]
	figpath = fpath.replace( '-fit.h5', '' )

	fd = h5py.File( fpath )
	dset_serial = fd['fit-bandwidth-latency-serial.dset']
	dset_mpi = fd['fit-bandwidth-latency-mpi.dset']
	dset_mpi_barrier = fd['fit-bandwidth-latency-mpi-barrier.dset']

	num_ngpus = dset_serial.shape[0]

	pobj = PlotFitBL( num_ngpus )
	pobj.savefig( dset_serial, 'serial', figpath+'/'+'fit-bl-avg-serial.png' )
	pobj.savefig( dset_mpi, 'mpi', figpath+'/'+'fit-bl-avg-mpi.png' )
	pobj.savefig( dset_mpi_barrier, 'mpi-barrier', figpath+'/'+'fit-bl-avg-mpi-barrier.png' )
