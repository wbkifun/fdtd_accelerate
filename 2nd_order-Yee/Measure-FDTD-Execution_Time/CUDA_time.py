#!/usr/bin/env python

# FDTD 3D dielectric
# 300x300x224 (922 MBytes with ch arrays)
# 1000 tstep
# GCC 4.3.3
# NVCC 3.0
# Ubuntu 9.04

import numpy as np

a = 'GeForce 9800 GTX+'
b = 'GeForce GTX 280'
c = 'GeForce GTX 480'
d = 'Tesla C1060'

ta = 97.69247
tb = 31.73120
tc = 17.21593
td = 49.84571

def hhmm(t, a, b):
	t = t*60
	a = int(a*100)
	b = int(b*100)

	x = a*t/b
	hh = x/3600
	mm = (x - hh*3600)/60

	x0 = 96*3600 + 3*60
	return '%.2d:%.2d (%d) %d times' % (hh, mm, x, x0/x)

print a, hhmm(44,ta,tb)
print c, hhmm(44,tc,tb)
print d, hhmm(44,td,tb)

'''
print a, hhmm(66,ta,td)
print b, hhmm(66,tb,td)
print c, hhmm(66,tc,td)
'''
