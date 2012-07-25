#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


def gaussian(tstep, sigma, t0, dt=1./2):
	return np.exp(- 0.5 * ( np.float64(tstep - t0) * dt / sigma )**2 )


def tfsf_field(fw, w_dt, m, n):
	x = 2 * np.sin(w_dt / 2)
	#k_dx = 2 * np.arcsin((-1 <= x) * (x <= 1) * x)
	k_dx = 2 * np.arcsin( np.complex128(x) )
	fxt = np.fft.irfft(fw[:] * np.exp(- 1j * k_dx[:] * m) * np.exp(- 1j * w_dt[:] * n))

	return fxt


# setup
nx = n = 600
tmax, tgap = 1024, 20
npml = 100
spt1, spt2 = 200, 400

t = np.arange(tmax, dtype=np.float64)

x_unit = 10		# nm
dx = 1.			# * x_unit
dt = dx / 2		# Courant factor S=0.5
wavelength = 30 * dx	# 300 nm
frequency = 1./wavelength

# allocate arrays
ez = np.zeros(n, dtype=np.float64)
hy = np.zeros(n, dtype=np.float64)

# prepare for tfsf
ft = gaussian(t, sigma=20, t0=200)
fw = np.fft.rfft(ft)
'''
w_dt = np.ones(tmax / 2 + 1, dtype=np.float64)
w_dt[:-1] = np.fft.fftfreq(tmax, dt)[:tmax/2]
w_dt[:] *= 2 * np.pi * dt
w_dt2 = np.linspace(0, 1, tmax / 2 + 1) * 2 * np.pi * dt
assert( np.linalg.norm(w_dt - w_dt2) == 0 )
'''
w_dt = np.linspace(0, 1, tmax / 2 + 1) * 2 * np.pi * dt

'''
tmp1 = tfsf_field(fw, w_dt, 0, 0)
tmp2 = tfsf_field(fw, w_dt, 0, 2)
print np.linalg.norm(ft - tmp1)
print np.linalg.norm(tmp1[:-2] - tmp2[2:])
'''

tfsf_hy1 = - tfsf_field(fw, w_dt, 0, 0)
tfsf_ez1 = tfsf_field(fw, w_dt, 0.5, 0.5)

ft_aux = np.load('tfsf_ft_ez.npy')
tfsf_aux_hy1 = np.load('tfsf_aux_hy1.npy')
tfsf_aux_ez1 = np.load('tfsf_aux_ez1.npy')

print np.linalg.norm(ft_aux[:] - ft[:])
print np.linalg.norm(tfsf_aux_hy1[:] - tfsf_hy1[:])
print np.linalg.norm(tfsf_aux_ez1[:] - tfsf_ez1[:])


import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rc('lines', linestyle='None', marker='p', markersize=2)
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.set_ylabel('f(t)')
ax2.set_ylabel('hy')
ax3.set_ylabel('ez')
ax3.set_xlabel('t')

t = np.arange(tmax, dtype=np.float64)

ax1.plot(t, ft, label='afp')
ax1.plot(t, ft_aux, label='aux')
ax1.legend()
ax2.plot(t, tfsf_hy1, label='afp')
ax2.plot(t, tfsf_aux_hy1, label='aux')
ax2.legend()
ax3.plot(t, tfsf_ez1, label='afp')
ax3.plot(t, tfsf_aux_ez1, label='aux')
ax3.legend()
plt.show()
