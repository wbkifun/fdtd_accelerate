#!/usr/bin/env python

from scipy import optimize
import numpy as np
import h5py as h5
import sys


def fit_bandwidth(nbyte, dts):
	x = nbyte
	t = dts

	fitfunc = lambda p, x: p * x
	errfunc = lambda p, x, y: fitfunc(p, x) - y

	p0 = np.array(1e3)
	p1, success = optimize.leastsq(errfunc, p0, args=(x, t))
	bandwidth = 1. / p1

	return bandwidth



# Main
nbyte = np.load('nbytes5_1.npy')
dts = np.load('dts5_1.npy')

bw = fit_bandwidth(nbyte, dts)
print('bandwidth: %g' % bw)


# Plot
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
xticks = range(8, 33)

mdts = dts * 1e3
fit_dt = lambda x, bw: (1./bw * x) * 1e3
p0 = ax1.plot(xticks, mdts, linestyle='None', color='k', marker='o')#, markersize=5)
p1 = ax1.plot(xticks, fit_dt(nbyte, bw), color='k')
ax1.set_xlabel(r'Size [$\times2^{16}$ nbyte]')
ax1.set_ylabel(r'Time [ms]')
ax1.set_xlim(7, 33)
ax1.set_ylim(mdts.min()*0.9, mdts.max()*1.1)
ax1.legend((p0, p1), ('Measure', 'Fitted'), loc='best', numpoints=1)
plt.savefig('measure.eps', dpi=150)
plt.show()
