#!/usr/bin/env python

import numpy as np

nsize = 127
nloop = 50

dt_range = np.zeros(nsize,'f')
for i in range(1,nsize+1):
	exec_time = np.load('./110-time_datas-npy/%.3dMByte.npy' % i)
	dt = exec_time[1:] - exec_time[:-1]

	for j in range(nloop-1):
		if( dt[j:].std() < 0.002 ):
			sj = j
			break
	dt_mean = dt[sj:].mean()
	print "[%.3d MByte] dt[%d:].mean()= %g" % (i, sj, dt_mean)
	dt_range[i-1] = dt_mean

print dt_range

from scipy import optimize
fitfunc = lambda p, x: p[0]*x + p[1]
errfunc = lambda p, x, y: fitfunc(p, x) - y
p0 = [1,0]
X = np.arange(1,nsize+1)
p1, success = optimize.leastsq(errfunc, p0[:], args=(X,dt_range))
print p1
chi2 = sum( pow(errfunc(p1, X, dt_range), 2) )
print chi2

import matplotlib.pyplot as pl
pl.plot(dt_range, 'o-')
pl.plot(p1[0]*X+p1[1], lw=3, color='red')
pl.show()
