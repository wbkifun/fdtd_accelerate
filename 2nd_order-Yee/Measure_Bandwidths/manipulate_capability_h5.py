#!/usr/bin/env python

import numpy as np
import h5py as h5

path = './capability_fdtd3d.h5'
f = h5.File(path, 'r')
tmax = f.attrs['tmax']

# cpu
cpu_name = f['cpu'].keys()[0]
cpu0 = f['cpu'][cpu_name]
nxs_cpu_4t = cpu0['nx_4T'].value
dts_cpu_4t = cpu0['dt_4T'].value

# gpu
gpu_name = f['gpu'].keys()[0]
gpu0 = f['gpu'][gpu_name]
nxs_gpu = gpu0['nx'].value
dts_gpu = gpu0['dt'].value

targets = ['cpu 4T', 'gpu']
nxs_list = [nxs_cpu_4t[3:], nxs_gpu[3:]]
dts_list = [dts_cpu_4t[3:], dts_gpu[3:]]

print('cpu name = %s' % cpu_name)
print('gpu name = %s' % gpu_name)
print('sample size = %s' % nxs_gpu.shape)


# Fitting
from scipy import optimize
fitfunc = lambda p, x: (p[0] * x + p[1]) * tmax
errfunc = lambda p, x, y: fitfunc(p, x) - y

cellss = []
latencys = []
p0 = np.array([1e-8, 0])
for nxs, dts in zip(nxs_list, dts_list):
	p1, success = optimize.leastsq(errfunc, p0, args=(nxs**3, dts))
	#p1 = optimize.fmin_slsqp(errfunc, p0, args=(nxs**3, dts))
	cellss.append(1 / p1[0])
	latencys.append(p1[1])


# Print
for target, cells, latency in zip(targets, cellss, latencys):
	print('%s: %1.2f MCells/s, latency = %g s' % (target, cells/1e6, latency))


# Plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks(nxs_gpu**3)
ax1.set_xticklabels([r'$%d^3$' % nx for nx in nxs_gpu])
ax1.set_xlabel('Cell Size')
ax1.set_ylabel('Time (s)')

for nxs, dts, cells, latency in zip(nxs_list, dts_list, cellss, latencys):
	ax1.plot(nxs**3, dts, linestyle='None', marker='p', markersize=4)
	
	p = (1./cells, latency)
	x = np.array([nxs[0], nxs[-1]]) ** 3
	y = fitfunc(p, x)
	ax1.plot(x, y)

plt.show()
