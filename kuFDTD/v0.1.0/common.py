'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 2009. 7. 23

 Copyright : GNU GPL
'''

import scipy as sc
import sys
from time import *


base_dir = '/usr/lib/python2.5/site-packages/kufdtd'


light_velocity = 2.99792458e8	# m s- 
ep0 = 8.85418781762038920e-12	# F m-1 (permittivity at vacuum)
mu0 = 1.25663706143591730e-6	# N A-2 (permeability at vacuum)
imp0 = sc.sqrt( mu0/ep0 )		# (impedance at vacuum)
pi = 3.14159265358979323846


def print_elapsed_time( t0, t1, tstep ):
	elapse_time = localtime(t1-t0-60*60*9)
	str_time = strftime('[%j]%H:%M:%S', elapse_time)
	print '%s    tstep = %d' % (str_time, tstep)


def list_replace( lst, index, content ):
	lst2 = []
	for element in lst:
		lst2.append( element )
	lst2.pop( index )
	lst2.insert( index, content )

	return lst2


def mem_human_unit( mem ):
	kbyte = 1024
	mbyte = 1024**2
	gbyte = 1024**3
	pbyte = 1024**4

	if mem >= pbyte: 
		return int( mem/pbyte ), 'PB'
	elif mem >= gbyte: 
		return int( mem/gbyte ), 'GB'
	elif mem >= mbyte: 
		return int( mem/mbyte ), 'MB'
	elif mem >= kbyte: 
		return int( mem/kbyte ), 'KB'
	else: 
		return int( mem ), 'B'


def print_mem_usage( mlist ):
	mhu = mem_human_unit

	total_list = []
	for i, ml in enumerate( mlist ): 
		total_list.append( sc.array( mlist[i] ).sum() )
	total = sc.array( total_list ).sum()	
	print 'total: %d %s' % mhu(total), '( %d %s,' % mhu(total_list[0]), '%d %s )' % mhu(total_list[1])

	if len(mlist[0]) > 1:
		for i in xrange( 0, len(mlist[0]) ):
			total = mlist[0][i] + mlist[1][i]
			print 'rank(%d):' % i, '%d %s' % mhu(total), '( %d %s,' % mhu(mlist[0][i]), '%d %s )' % mhu(mlist[1][i])
