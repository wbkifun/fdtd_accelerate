#!/usr/bin/env python

from scipy import optimize
import numpy as np
import h5py as h5
import sys


def fit_bandwidth_latency(nbyte, dts):
	x = nbyte
	t = dts

	fitfunc = lambda p, x: p[0] * x + p[1]
	errfunc = lambda p, x, y: fitfunc(p, x) - y

	p0 = np.array([1e3, 0])
	p1, success = optimize.leastsq(errfunc, p0, args=(x, t))
	bandwidth = 1. / p1[0]
	latency = p1[1]

	return (bandwidth, latency)



# Main
nbyte = np.load('nbytes5_1.npy')
#dts = np.load('dts5_1.npy') * 1.2       # memcpy
dts = np.load('dts5_1.npy') * 0.93     # mpi

bw, lt = fit_bandwidth_latency(nbyte, dts)
print('bandwidth: %g' % bw)
print('latency: %g' % lt)

'''
# Plot
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
xticks = range(8, 33)

mdts = dts * 1e3
fit_dt = lambda x, bw, lt: (1./bw * x + lt) * 1e3
p0 = ax1.plot(xticks, mdts, linestyle='None', color='k', marker='o')#, markersize=5)
p1 = ax1.plot(xticks, fit_dt(nbyte, bw, lt), color='k')
ax1.set_xlabel(r'Size [$\times2^{16}$ nbyte]')
ax1.set_ylabel(r'Time [ms]')
ax1.set_xlim(7, 33)
ax1.set_ylim(mdts.min()*0.9, mdts.max()*1.1)
ax1.legend((p0, p1), ('Measure', 'Fitted'), loc='best', numpoints=1)
plt.savefig('measure.eps', dpi=150)
plt.show()
'''
