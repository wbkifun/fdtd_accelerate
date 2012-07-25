#!/usr/bin/env python

from scipy import optimize
import numpy as np
import h5py as h5
import sys


def fit_bandwidth_latency(nbytes, pre_bandwidth):
	x = nbytes
	t = 1. / (pre_bandwidth * 1024**2) * nbytes	# MByte/s

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
for key, obj in f.items():
	if key == 'nbytes':
		pass
	elif key == 'dtod':
		fitted[key] = fit_bandwidth_latency(nbytes, obj.value)
		#print(key, fit_bandwidth_latency(nbytes, obj.value))
	else:
		fitted[key] = {}
		for key2, obj2 in obj.items():
			fitted[key][key2] = fit_bandwidth_latency(nbytes, obj2.value)
			#print(key, key2, fit_bandwidth_latency(nbytes, obj2.value))

for key, val in fitted.items():
	if key == 'dtod':
		print('%s\t%g\t%g' % (key, val[0], val[1]))
	else:
		for key2, val2 in val.items():
			print('%s,%s\t%g\t%g' % (key, key2 , val2[0], val2[1]))


# Plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

raw_dt = lambda x, bw: 1./(bw * 1024**2) * x
fit_dt = lambda x, bw, lt: 1./bw * x + lt
plots = []
legends = []
for key, val in fitted.items():
	if key == 'dtod':
		ax1.plot(nbytes, raw_dt(nbytes, f[key].value), linestyle='None', marker='p', markersize=4)
		plots.extend( ax1.plot(nbytes, fit_dt(nbytes, val[0], val[1])) )
		legends.append(key)
	else:
		for key2, val2 in val.items():
			ax1.plot(nbytes, raw_dt(nbytes, f[key][key2].value), linestyle='None', marker='p', markersize=4)
			plots.extend( ax1.plot(nbytes, fit_dt(nbytes, val2[0], val2[1])) )
			legends.append(key + ',' + key2)
plt.legend(plots, legends, loc='upper left')
plt.savefig(h5_path.replace('.h5', '.png'), dpi=150)
#plt.show()
