#!/usr/bin/env python

import scipy as sc

x0 = 200
sigma = 100
Nx = 500

a = sc.zeros(Nx, 'f')

for i in xrange(Nx):
	a[i] = sc.exp( -0.5*(i-x0)**2/sigma )

import pylab as pl
pl.plot(a)
pl.show()
