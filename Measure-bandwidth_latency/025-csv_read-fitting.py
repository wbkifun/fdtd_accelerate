#!/usr/bin/env python

from pylab import *
import sys

gtime = zeros((10,30), 'f')

filename = './profiler-csv/020-C1060-120MB-30iter-%.2d.csv' % 1

f = open(filename)
j = 0
for line in f.readlines():
	if 'func' in line:
		print j, line
		gtime[0,j] = float64( line.split(',')[2] )
		j += 1
f.close()

print gtime[0,:]
dt = gtime[:,1:] - gtime[:,:-1]
print dt[0,:]

'''
for i in range(10):
	filename = './profiler-csv/020-C1060-120MB-30iter-%.2d.csv' % (i+1)

	f = open(filename)
	j = 0
	for line in f.readlines():
		if 'func' in line:
			#print j, line
			gtime[i,j] = float64( line.split(',')[2] )
			j += 1
	f.close()

print gtime
dt = gtime[:,1:] - gtime[:,:-1]
print dt
print dt.shape
'''
'''
gtime_mean = gtime.mean(axis=1)
dt = gtime_mean[1:] - gtime_mean[:-1]
print dt
print dt.shape
'''
