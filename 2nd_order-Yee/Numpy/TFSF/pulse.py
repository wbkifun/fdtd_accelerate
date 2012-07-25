#!/usr/bin/env python

import numpy as np


def gaussian(tstep, sigma, t0, dt=1./2):
	return np.exp(- 0.5 * ( np.float64(tstep - t0) * dt / sigma )**2 )


def ft2fxt(ft, tmax, x, t, dt=1./2):
	fw = np.fft.rfft(ft)
	#w = np.fft.fftfreq(tmax, 0.5) * 2 * np.pi
	w = np.linspace(0, 1, tmax/2 + 1) * 2 * np.pi
	
	if type(x) == np.ndarray:
		fxt = np.zeros((x.size, tmax), dtype=np.float64)
		for i in x:
			fxt[i,:] = np.fft.irfft(fw[:] * np.exp(- 1j * w[:] * i) * np.exp(1j * w[:] * dt * t))
	else:
		fxt = np.fft.irfft(fw[:] * np.exp(- 1j * w[:] * x) * np.exp(1j * w[:] * dt * t))

	return fxt


nx = 500
tmax = 1024
x = np.arange(nx, dtype=np.float64)
t = np.arange(tmax, dtype=np.float64)

ft = gaussian(t, sigma=20, t0=200)
fxt = ft2fxt(ft, tmax, x, t=0)

tmp1 = ft2fxt(ft, tmax, 0, 0)
tmp2 = ft2fxt(ft, tmax, 0, -10)
print np.linalg.norm(ft - tmp1)
print np.linalg.norm(tmp1[:-10] - tmp2[10:])


fxt2 = ft2fxt(ft, tmax, x, t=4)
print np.linalg.norm(fxt - fxt2)
print np.linalg.norm(fxt[:-2] - fxt2[2:])
#assert( np.linalg.norm(fxt - fxt2) == 0 )

import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rc('lines', linestyle='None', marker='p', markersize=1)
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.set_xlabel('t')
ax1.set_ylabel('f(t)')
#ax2.set_xlabel('x')
#ax2.set_ylabel('f(x,t)')

ax1.plot(t, ft)
#ax2.plot(x, fxt[:,0], x, fxt2[:,0])
ax2.plot(t, tmp1[:], label='origin')
ax2.plot(t, tmp2[:], label='shift')
ax2.legend()
plt.show()
'''
for tstep in xrange(0, tmax, 10):
	line2.set_ydata(fxt[:,tstep])
	plt.draw()
'''
