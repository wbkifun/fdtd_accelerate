#!/usr/bin/env python

from scipy import optimize
import numpy as np
import h5py as h5
import sys


def fit_bandwidth_latency(nbytes, dts):
	x = nbytes
	t = dts

	fitfunc = lambda p, x: p[0] * x + p[1]
	errfunc = lambda p, x, y: fitfunc(p, x) - y

	p0 = np.array([1e-6, 0])
	p1, success = optimize.leastsq(errfunc, p0, args=(x, t))
	bandwidth = 1. / p1[0]
	latency = p1[1]

	return (bandwidth, latency)


# Main
try:
	h5_path = sys.argv[1]
except IndexError:
	print('Error : h5 file required!')
	sys.exit()

f = h5.File(h5_path, 'r')
fitted = {}
nbytes = f['nbytes'].value
dts = f['dts'].value
bw, lt = fit_bandwidth_latency(nbytes, dts)
print('mpi\t%g\t%g' % (bw, lt))


# Plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

fit_dt = lambda x, bw, lt: 1./bw * x + lt
ax1.plot(nbytes, dts, linestyle='None', marker='p', markersize=4)
ax1.plot(nbytes, fit_dt(nbytes, bw, lt))
plt.savefig(h5_path.replace('.h5', '.png'), dpi=150)
#plt.show()
