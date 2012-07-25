#!/usr/bin/env python

import numpy as np

dts = np.load('330.npy')
tmax = 1000
nxs = np.arange(96, 480+1, 32)	# nx**2

nxs = nxs[3:]
dts = dts[3:]


# Fitting
from scipy import optimize
fitfunc = lambda p, x: (p[0] * x + p[1]) * tmax
errfunc = lambda p, x, y: fitfunc(p, x) - y

p0 = np.array([1e-6, 0])
p1, success = optimize.leastsq(errfunc, p0, args=(nxs**2, dts))
cells = 1. / p1[0]
latency = p1[1]
print('%1.2f MCells/s, latency = %g s' % (cells/1e6, latency))


# Plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks(nxs**2)
ax1.set_xticklabels([r'$%d^2$' % nx for nx in nxs])
ax1.set_xlabel('Cell Size')
ax1.set_ylabel('Time (s)')

ax1.plot(nxs**2, dts, linestyle='None', marker='p', markersize=4)

x = np.array([nxs[0], nxs[-1]]) ** 2
y = fitfunc(p1, x)
ax1.plot(x, y)

plt.show()
