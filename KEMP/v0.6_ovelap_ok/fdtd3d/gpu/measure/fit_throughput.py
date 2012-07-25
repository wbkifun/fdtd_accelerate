#!/usr/bin/env python

from scipy import optimize
import numpy as np
import h5py as h5
import sys


def fit_throughput(point, dts):
	x = point
	t = dts

	fitfunc = lambda p, x: p * x
	errfunc = lambda p, x, y: fitfunc(p, x) - y

	p0 = np.array(1e3)
	p1, success = optimize.leastsq(errfunc, p0, args=(x, t))
	throughput = 1. / p1

	return throughput



# Main
point = np.load('measure_point.npy')
dts = np.load('measure_dts.npy')

thp = fit_throughput(point, dts)
print('throughput: %g' % thp)


# Plot
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
xticks = range(1, 26)

mdts = dts * 1e3
fit_dt = lambda x, thp: 1./thp * x * 1e3
p0 = ax1.plot(xticks, mdts, linestyle='None', color='k', marker='o')#, markersize=5)
p1 = ax1.plot(xticks, fit_dt(point, thp), color='k')
ax1.set_xlabel(r'Grid size [$\times2^{20}$ point]')
ax1.set_ylabel(r'Time [ms]')
ax1.set_xlim(0, 26)
#ax1.set_ylim(mdts.min()*0.9, mdts.max()*1.1)
ax1.legend((p0, p1), ('Measure', 'Fitted'), loc='best', numpoints=1)
plt.savefig('measure.eps', dpi=150)
plt.show()
