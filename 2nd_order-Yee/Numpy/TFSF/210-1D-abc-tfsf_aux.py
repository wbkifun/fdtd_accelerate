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

		if kwargs.has_key('src'):
			s.src_field = getattr(s, kwargs['src_field'])
			s.src_pt = int(kwargs['src_pt'])

			if kwargs['src'] == 'sin':
				s.src_omega = np.pi * float(kwargs['src_frequency']) * s.dt
				update_src = s.update_src_sin

			if kwargs['src'] == 'gaussian':
				s.src_t0 = int(kwargs['src_t0'])
				s.src_sigma = float(kwargs['src_sigma'])
				update_src = s.update_src_gaussian

			if 'e' in kwargs['src_field']:
				s.e_tfunc_list.append(update_src)
			elif 'h' in kwargs['src_field']:
				s.h_tfunc_list.append(update_src)

		if kwargs.has_key('src_tfsf'):
			tfsf_n = 4
			s.spt1, s.spt2 = int(kwargs['src_pt']), None
			if kwargs.has_key('src_pt2'):
				s.spt2 = int(kwargs['src_pt2'])
				tfsf_n += spt2 - spt1

			kwargs_aux = dict(src=kwargs['src_tfsf'])
			if kwargs['src_tfsf'] == 'sin':
				kwargs_aux['src_frequency'] = kwargs['src_frequency']

			if kwargs['src_tfsf'] == 'gaussian':
				kwargs_aux['src_t0'] = kwargs['src_t0']
				kwargs_aux['src_sigma'] = kwargs['src_sigma']

			s.fdtd_aux = fdtd1d(tfsf_n, abc='-+', src_field='ez', src_pt=1, **kwargs_aux)
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


	def update_src_sin(s, tstep):
		s.src_field[s.src_pt] += np.sin(s.src_omega * tstep)


	def update_src_gaussian(s, tstep):
		s.src_field[s.src_pt] += np.exp(- 0.5 * ( float(tstep - s.src_t0) * s.dt / s.src_sigma )**2 )


	def update_tfsf_e(s, tstep):
		s.fdtd_aux.update_h(tstep)
		s.ez[s.spt1] -= s.cez[s.spt1] * s.fdtd_aux.hy[2]
		if s.spt2 != None: 
			s.ez[s.spt2] += s.cez[s.spt2] * s.fdtd_aux.hy[-2]


	def update_tfsf_h(s, tstep):
		s.fdtd_aux.update_e(tstep)
		s.hy[s.spt1] -= 0.5 * s.fdtd_aux.ez[2]
		if s.spt2 != None:
			s.hy[s.spt2] += 0.5 * s.fdtd_aux.ez[-2]



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

#fdtd = fdtd1d(nx, abc='-', src_tfsf='sin', src_pt=spt1, src_frequency=1./50)
fdtd = fdtd1d(nx, abc='-', src_tfsf='gaussian', src_pt=spt1, src_pt2=spt2, src_t0=200, src_sigma=20)

# for plot
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure() #plt.figure(figsize=(20,7))
x = np.arange(n)
x2 = np.arange(spt1)
x3 = np.arange(spt2,nx)
min1, min2, max1, max2 = 0, 0, 0, 0

ax1 = fig.add_subplot(2,1,1)
line1, = ax1.plot(x, np.zeros(n, dtype=np.float32), linestyle='None', marker='p', markersize=2)
ax1.set_xlim(0, nx)
ax1.set_ylim(-1.2, 1.2)
from matplotlib.patches import Rectangle
rect = Rectangle((spt1, -1.2), spt2-spt1, 2.4, facecolor='w', linestyle='dashed')
ax1.add_patch(rect)

ax2 = fig.add_subplot(2,2,3)
line2, = ax2.plot(x2, np.zeros(spt1, dtype=np.float32), linestyle='None', marker='p', markersize=2)
ax2.set_xlim(0, spt1)

ax3 = fig.add_subplot(2,2,4)
line3, = ax3.plot(x3, np.zeros(nx-spt2, dtype=np.float32), linestyle='None', marker='p', markersize=2)
ax3.set_xlim(spt2, nx)


# main loop
for tstep in xrange(tmax):
	fdtd.update_e(tstep)

	fdtd.update_h(tstep)

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
		plt.draw()

print ''
plt.show()
