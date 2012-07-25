#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


class fdtd1d:
	def __init__(s, nx, **kwargs):
		s.nx = s.n = nx
		s.dt = 0.5
		s.eh_fields = s.ez, s.hy = [np.zeros(s.n, dtype=np.float32) for i in range(2)]
		s.cez = np.ones(s.n, dtype=np.float32) * 0.5

		s.e_func_list = [s.update_main_e]
		s.e_tfunc_list = []
		s.h_func_list = [s.update_main_h]
		s.h_tfunc_list = []

		if kwargs.has_key('abc'):
			if '+' in kwargs['abc']:
				s.abc_ez = np.zeros(2, dtype=np.float32)
				s.e_func_list.append(s.update_abc_e)

			if '-' in kwargs['abc']:
				s.abc_hy = np.zeros(2, dtype=np.float32)
				s.h_func_list.append(s.update_abc_h)

		if kwargs.has_key('src_tfsf'):
			s.spt1, s.spt2 = int(kwargs['src_pt']), None
			if kwargs.has_key('src_pt2') and kwargs['src_pt2'] != None:
				s.spt2 = int(kwargs['src_pt2'])

			if kwargs['src_tfsf'] == 'gaussian':
				t0 = int(kwargs['src_t0'])
				sigma = float(kwargs['src_sigma'])
				s.prepare_tfsf_gaussian(t0, sigma)

			s.e_tfunc_list.append(s.update_tfsf_e)
			s.h_tfunc_list.append(s.update_tfsf_h)


	def update_e(s, tstep=None):
		for e_func in s.e_func_list: e_func()
		for e_tfunc in s.e_tfunc_list: e_tfunc(tstep)


	def update_h(s, tstep=None):
		for h_func in s.h_func_list: h_func()
		for h_tfunc in s.h_tfunc_list: h_tfunc(tstep)


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


	def update_tfsf_e(s, tstep):
		s.ez[s.spt1] -= s.cez[s.spt1] * s.tfsf_hy[tstep]
		if s.spt2 != None: 
			s.ez[s.spt2] += s.cez[s.spt2] * s.tfsf_hy2[tstep]


	def update_tfsf_h(s, tstep):
		s.hy[s.spt1] -= 0.5 * s.tfsf_ez[tstep]
		if s.spt2 != None:
			s.hy[s.spt2] += 0.5 * s.tfsf_ez2[tstep]


	def get_tfsf_inc(s, fw, w_dt, m, n):
		x = 2 * np.sin(w_dt / 2)
		k_dx = 2 * np.arcsin((-1 <= x) * (x <= 1) * x)
		return np.fft.irfft(fw[:] * np.exp(- 1j * k_dx[:] * m) * np.exp(- 1j * w_dt[:] * n))


	def prepare_tfsf_gaussian(s, t0, sigma):
		t = np.arange(tmax, dtype=np.float32)
		ft = np.exp(- 0.5 * ( (t - t0) * s.dt / sigma )**2 )
		fw = np.fft.rfft(ft)
		w_dt = np.linspace(0, 1, tmax / 2 + 1) * 2 * np.pi * dt

		s.tfsf_hy = - s.get_tfsf_inc(fw, w_dt, 0, 0)
		s.tfsf_ez = s.get_tfsf_inc(fw, w_dt, 0.5, 0.5)

		if s.spt2 != None: 
			s.tfsf_hy2 = - s.get_tfsf_inc(fw, w_dt, (s.spt2 - s.spt1), 0)
			s.tfsf_ez2 = s.get_tfsf_inc(fw, w_dt, (s.spt2 - s.spt1) + 0.5, 0.5)



# setup
nx = n = 400
tmax, tgap = 1024, 20
spt1, spt2 = 100, 300

x_unit = 10		# nm
dx = 1.			# * x_unit
dt = dx / 2		# Courant factor S=0.5
wavelength = 50 * dx	# 300 nm
frequency = 1./wavelength
w_dt = (2 * np.pi * frequency) * dt

fdtd = fdtd1d(nx, abc='-', src_tfsf='gaussian', src_pt=spt1, src_pt2=spt2, src_t0=200, src_sigma=20)

# prepare for plot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
mpl.rc('lines', linestyle='None', marker='p', markersize=2)

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,2,5)
ax4 = fig.add_subplot(3,2,6)

t = np.arange(tmax)
x = np.arange(nx)
x1 = np.arange(spt1)
x2 = np.arange(spt2, nx)
min1, min2, max1, max2 = 0, 0, 0, 0

line1, = ax1.plot(t, fdtd.tfsf_ez)
ax1.set_xlim(0, tmax)
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel('t')
ax1.set_ylabel('f(t)')

line2, = ax2.plot(x, fdtd.ez)
ax2.set_xlim(0, nx)
ax2.set_ylim(-1.2, 1.2)
ax2.set_xlabel('x')
ax2.set_ylabel('ez')
rect_tfsf = Rectangle((spt1, -1.2), spt2-spt1, 2.4, facecolor='w', linestyle='dashed')
ax2.add_patch(rect_tfsf)
#rect_pml = Rectangle((nx-npml, -1.5), npml, 3, alpha=0.1)
#ax2.add_patch(rect_pml)

line3, = ax3.plot(x1, fdtd.ez[:spt1])
ax3.set_xlim(0, spt1)
ax3.set_xlabel('x')
ax3.set_ylabel('ez')

line4, = ax4.plot(x2, fdtd.ez[spt2:])
ax4.set_xlim(spt2, nx)
ax4.set_xlabel('x')


# main loop
for tstep in xrange(tmax):
	fdtd.update_h(tstep)

	fdtd.update_e(tstep)


	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()

		line2.set_ydata(fdtd.ez[:])
		min1 = min(min1, fdtd.ez[:spt1].min())
		max1 = max(max1, fdtd.ez[:spt1].max())
		min2 = min(min2, fdtd.ez[spt2:].min())
		max2 = max(max2, fdtd.ez[spt2:].max())
		ax3.set_ylim(min1, max1)
		line3.set_ydata(fdtd.ez[:spt1])
		ax4.set_ylim(min2, max2)
		line4.set_ydata(fdtd.ez[spt2:])
		plt.draw()

print ''
plt.show()
