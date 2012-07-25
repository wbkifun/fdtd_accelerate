#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def gaussian(tstep, sigma, t0=0):
	return np.exp(- 0.5 * ( np.float32(tstep - t0) / sigma )**2 )


def numeric_k_dx(w_dt):
	return 2 * np.arcsin(2 * np.sin(w_dt / 2))


def tfsf_field(ft, dt, m):
	fw = np.fft.rfft(ft)
	w_dt = np.fft.fftfreq(tmax, dt)[:tmax/2] * 2 * np.pi * dt
	k_dx = numeric_k_dx(w_dt)
	
	return np.fft.irfft(fw * np.exp(1j * k_dx * m))


dx = 1.
dt = dx / 2
tmax = 1000
tstep = np.arange(1,tmax+1)
ft = gaussian(tstep, sigma=30, t0=150)

fw = np.fft.rfft(ft)

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
line1, = ax1.plot(tstep, ft) 
ax2 = fig.add_subplot(2,1,2)
line2, = ax2.plot(np.abs(fw)[:int((tmax/2)/30)])

plt.show()
