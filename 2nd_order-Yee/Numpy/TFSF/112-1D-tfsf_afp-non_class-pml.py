#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


def gaussian(tstep, sigma, t0, dt=1./2):
	return np.exp(- 0.5 * ( np.float32(tstep - t0) * dt / sigma )**2 )


def tfsf_field(fw, w_dt, m, n):
	x = 2 * np.sin(w_dt / 2)
	k_dx = 2 * np.arcsin((-1 <= x) * (x <= 1) * x)
	fxt = np.fft.irfft(fw[:] * np.exp(- 1j * k_dx[:] * m) * np.exp(- 1j * w_dt[:] * n))

	return fxt


# setup
nx = n = 600
tmax, tgap = 2048, 20
npml = 100
spt1, spt2 = 200, 400

x_unit = 10		# nm
dx = 1.			# * x_unit
dt = dx / 2		# Courant factor S=0.5
wavelength = 30 * dx	# 300 nm
frequency = 1./wavelength

# allocate arrays
ez = np.zeros(n, dtype=np.float32)
hy = np.zeros(n, dtype=np.float32)

# prepare for tfsf
ft = gaussian(np.arange(tmax, dtype=np.float32), sigma=20, t0=200)
fw = np.fft.rfft(ft)
'''
w_dt = np.ones(tmax / 2 + 1, dtype=np.float32)
w_dt[:-1] = np.fft.fftfreq(tmax, dt)[:tmax/2]
w_dt[:] *= 2 * np.pi * dt
w_dt2 = np.linspace(0, 1, tmax / 2 + 1) * 2 * np.pi * dt
assert( np.linalg.norm(w_dt - w_dt2) == 0 )
'''
w_dt = np.linspace(0, 1, tmax / 2 + 1) * 2 * np.pi * dt
print w_dt.dtype

tfsf_hy1 = - tfsf_field(fw, w_dt, 0, 0)
tfsf_ez1 = tfsf_field(fw, w_dt, 0.5, 0.5)
tfsf_hy2 = - tfsf_field(fw, w_dt, (spt2 - spt1), 0)
tfsf_ez2 = tfsf_field(fw, w_dt, (spt2 - spt1) + 0.5, 0.5)

# prepare pml
m = 4
print (m+1.)/(15*np.pi*npml)*0.5/C
#sigma_dt = ( (np.linspace(1,npml,npml)/npml)**m )*(m+1.)/(15*np.pi*npml)*0.5/C
sigma_dt = ( (np.linspace(1,npml,npml)/npml)**m )*0.1
#sigma_dt = np.ones(npml, dtype=np.float32)*0.01
pml_ca = (2. - sigma_dt[:])/(2 + sigma_dt[:])
pml_cb = 1./(2 + sigma_dt[:])

# prepare for plot
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('lines', linestyle='None', marker='p', markersize=2)

from matplotlib.patches import Rectangle

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

line1, = ax1.plot(t, ft)
ax1.set_xlim(0, tmax)
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel('t')
ax1.set_ylabel('f(t)')

line2, = ax2.plot(x, ez)
ax2.set_xlim(0, nx)
ax2.set_ylim(-1.2, 1.2)
ax2.set_xlabel('x')
ax2.set_ylabel('ez')
rect_tfsf = Rectangle((spt1, -1.2), spt2-spt1, 2.4, facecolor='w', linestyle='dashed')
ax2.add_patch(rect_tfsf)
rect_pml = Rectangle((nx-npml, -1.5), npml, 3, alpha=0.1)
ax2.add_patch(rect_pml)

line3, = ax3.plot(x1, ez[:spt1])
ax3.set_xlim(0, spt1)
ax3.set_xlabel('x')
ax3.set_ylabel('ez')

line4, = ax4.plot(x2, ez[spt2:])
ax4.set_xlim(spt2, nx)
ax4.set_xlabel('x')

# main loop
for tstep in xrange(tmax):
	hy[1:] -= 0.5 * (- ez[1:] + ez[:-1])
	#hy[1:-npml] -= 0.5 * (- ez[1:-npml] + ez[:-npml-1])

	# tfsf
	hy[spt1] -= 0.5 * tfsf_ez1[tstep]
	#hy[spt2] += 0.5 * tfsf_ez2[tstep]

	# pml
	#hy[-npml:] = pml_ca[:] * hy[-npml:] - pml_cb[:] * (- ez[-npml:] + ez[-npml-1:-1]) 


	ez[:-1] += 0.5 * (hy[1:] - hy[:-1])
	#ez[:-npml-1] += 0.5 * (hy[1:-npml] - hy[:-npml-1])

	# tfsf
	ez[spt1] -= 0.5 * tfsf_hy1[tstep]
	#ez[spt2] += 0.5 * tfsf_hy2[tstep]

	# pml
	#ez[-npml-1:-1] = pml_ca[:] * ez[-npml-1:-1] + pml_cb[:] * (hy[-npml:] - hy[-npml-1:-1])


	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()

		line2.set_ydata(ez[:])
		min1 = min(min1, ez[:spt1].min())
		max1 = max(max1, ez[:spt1].max())
		min2 = min(min2, ez[spt2:].min())
		max2 = max(max2, ez[spt2:].max())
		ax3.set_ylim(min1, max1)
		line3.set_ydata(ez[:spt1])
		ax4.set_ylim(min2, max2)
		line4.set_ydata(ez[spt2:])
		plt.draw()

print ''
plt.show()
