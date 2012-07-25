#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


class fdtd1d:
	def __init__(s, nx, **kargs):
		s.nx = s.n = nx
		s.eh_fields = s.ez, s.hy = [np.zeros(s.n, dtype=np.float64) for i in range(2)]
		s.cez = np.ones(s.n, dtype=np.float64) * 0.5

		s.update_e_func_list = [s.update_main_e]
		s.update_h_func_list = [s.update_main_h]

		if kargs['abc'] == 'on':
			s.abc_ez = np.zeros(2, dtype=np.float64)
			s.abc_hy = np.zeros(2, dtype=np.float64)
			s.update_e_func_list.append(s.update_abc_e)
			s.update_h_func_list.append(s.update_abc_h)


	def update_e(s):
		for update_e_func in s.update_e_func_list:
			update_e_func()


	def update_h(s):
		for update_h_func in s.update_h_func_list:
			update_h_func()


	def update_main_e(s):
		s.ez[:-1] += s.cez[:-1] * (s.hy[1:] - s.hy[:-1])


	def update_main_h(s):
		s.hy[1:] += 0.5 * (s.ez[1:] - s.ez[:-1])


	def update_abc_e(s):
		s.ez[-1] = s.abc_ez[-1]
		s.abc_ez[-1] = s.abc_ez[-2]
		s.abc_ez[-2] = s.ez[-2]


	def update_abc_h(s):
		s.hy[0] = s.abc_hy[0]
		s.abc_hy[0] = s.abc_hy[1]
		s.abc_hy[1] = s.hy[1]


	def update_tfsf_e(s, spt1, spt2, fdtd_aux):
		s.ez[spt1] -= s.cez[spt1] * fdtd_aux.hy[2]
		#s.ez[spt2] += s.cez[spt2] * fdtd_aux.hy[-2]


	def update_tfsf_h(s, spt1, spt2, fdtd_aux):
		s.hy[spt1] -= 0.5 * fdtd_aux.ez[2]
		#s.hy[spt2] += 0.5 * fdtd_aux.ez[-2]


	def update_src_e(s, tstep, spt, **kargs):
		if kargs['ft'] == 'sin':
			s.ez[spt] += np.sin(np.pi * kargs['frequency'] * tstep)

		elif kargs['ft'] == 'gaussian':
			s.ez[spt] += np.exp(- 0.5 * ( np.float64(tstep - kargs['t0']) / kargs['sigma'] )**2 )



# setup
nx = n = 400
tmax, tgap = 1024, 10
spt1, spt2 = 100, 200

x_unit = 10		# nm
dx = 1.			# * x_unit
dt = dx / 2		# Courant factor S=0.5
wavelength = 50 * dx	# 300 nm
frequency = 1./wavelength
w_dt = (2 * np.pi * frequency) * dt

fdtd = fdtd1d(nx, abc='on')
naux = spt2-spt1+4
#naux = 100
fdtd_aux = fdtd1d(naux, abc='on')


# for plot
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure() #plt.figure(figsize=(20,7))
x = np.arange(n)
x2 = np.arange(spt1)
x3 = np.arange(spt2,nx)
x4 = np.arange(naux)
min1, min2, max1, max2 = 0, 0, 0, 0

ax1 = fig.add_subplot(3,1,1)
line1, = ax1.plot(x, np.zeros(n, dtype=np.float64), linestyle='None', marker='p', markersize=2)
ax1.set_xlim(0, nx)
ax1.set_ylim(-1.2, 1.2)
from matplotlib.patches import Rectangle
rect = Rectangle((spt1, -1.2), spt2-spt1, 2.4, facecolor='w', linestyle='dashed')
ax1.add_patch(rect)

ax2 = fig.add_subplot(3,2,3)
line2, = ax2.plot(x2, np.zeros(spt1, dtype=np.float64), linestyle='None', marker='p', markersize=2)
ax2.set_xlim(0, spt1)

ax3 = fig.add_subplot(3,2,4)
line3, = ax3.plot(x3, np.zeros(nx-spt2, dtype=np.float64), linestyle='None', marker='p', markersize=2)
ax3.set_xlim(spt2, nx)

ax4 = fig.add_subplot(3,1,3)
line4, = ax4.plot(x4, np.zeros(naux, dtype=np.float64), linestyle='None', marker='p', markersize=2)
ax4.set_xlim(0, naux)
ax4.set_ylim(-1e-6, 1e-6)

# main loop
for tstep in xrange(tmax):
	fdtd.update_e()
	fdtd.update_tfsf_e(spt1, spt2, fdtd_aux)
	fdtd_aux.update_e()
	#fdtd_aux.update_src_e(tstep, 1, ft='sin', frequency=1./50)
	fdtd_aux.update_src_e(tstep, 1, ft='gaussian', t0=150, sigma=20)

	fdtd.update_h()
	fdtd.update_tfsf_h(spt1, spt2, fdtd_aux)
	fdtd_aux.update_h()

	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()
		line1.set_ydata(fdtd.ez[:])

		min1 = min(min1, fdtd.ez[:spt1].min())
		max1 = max(max1, fdtd.ez[:spt1].max())
		min2 = min(min1, fdtd.ez[spt2:].min())
		max2 = max(max2, fdtd.ez[spt2:].max())
		ax2.set_ylim(min1, max1)
		line2.set_ydata(fdtd.ez[:spt1])
		ax3.set_ylim(min2, max2)
		line3.set_ydata(fdtd.ez[spt2:])

		line4.set_ydata(fdtd_aux.ez[:])
		plt.draw()

print ''
plt.show()
