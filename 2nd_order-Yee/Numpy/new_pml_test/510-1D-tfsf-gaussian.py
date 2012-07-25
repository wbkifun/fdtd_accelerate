#!/usr/bin/env python

from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0
import numpy as np
import sys


def gaussian(tstep, sigma, t0=0):
	return np.exp(- 0.5 * ( np.float32(tstep - t0) / sigma )**2 )


def numeric_k_dx(w_dt):
	x = 2 * np.sin(w_dt / 2)
	xx = (-1 <= x) * (x <= 1) * x
	return 2 * np.arcsin(xx)


def tfsf_field(fw, w_dt, m):
	k_dx = numeric_k_dx(w_dt)
	
	return np.fft.irfft(fw[:] * np.exp(1j * k_dx[:] * m))


# setup
nx = n = 600
tmax, tgap = 1024, 20
npml = 100
src1, src2 = 200, 400

x_unit = 10		# nm
dx = 1.			# * x_unit
dt = dx / 2		# Courant factor S=0.5
wavelength = 30 * dx	# 300 nm
frequency = 1./wavelength

# allocate arrays
ez = np.zeros(n, 'f')
hy = np.zeros(n, 'f')

# prepare for tfsf
ft = gaussian(np.arange(tmax, dtype=np.float32), sigma=30, t0=150)
fw = np.fft.rfft(ft)
w_dt = np.ones(tmax / 2 + 1, dtype=np.float32)
w_dt[:-1] = np.fft.fftfreq(tmax, dt)[:tmax/2]
w_dt[:] *= 2 * np.pi * dt
#w_dt = np.linspace(0, 1, tmax / 2 + 1) * 2 * np.pi
tfsf_hy1 = tfsf_field(fw, w_dt, 0)
tfsf_ez1 = tfsf_field(fw, w_dt, 0.5)
tfsf_hy2 = tfsf_field(fw, w_dt, (src2-src1))
tfsf_ez2 = tfsf_field(fw, w_dt, (src2-src1)+0.5)

tmp = np.zeros(tmax, 'f')
tmp[:-200] = tfsf_hy2[200:]
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(tfsf_hy1) 
ax1.plot(tmp) 
ax2 = fig.add_subplot(2,1,2)
ax2.plot(tfsf_hy2)
plt.show()
'''
# prepare pml
m = 4
print (m+1.)/(15*np.pi*npml)*0.5/C
#sigma_dt = ( (np.linspace(1,npml,npml)/npml)**m )*(m+1.)/(15*np.pi*npml)*0.5/C
sigma_dt = ( (np.linspace(1,npml,npml)/npml)**m )*0.1
#sigma_dt = np.ones(npml, 'f')*0.01
pml_ca = (2. - sigma_dt[:])/(2 + sigma_dt[:])
pml_cb = 1./(2 + sigma_dt[:])

# prepare for plot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
rect = Rectangle((nx-npml, -1.5), npml, 3, alpha=0.1)
plt.gca().add_patch(rect)

#plt.figure(figsize=(20,7))
plt.ion()
x = np.arange(n)
line, = plt.plot(x, np.ones(n, 'f'))
plt.axis([0,n,-1.5,1.5])

# main loop
for tstep in xrange(tmax):
	ez[:-1] += 0.5 * (hy[1:] - hy[:-1])
	#ez[:-npml-1] += 0.5 * (hy[1:-npml] - hy[:-npml-1])

	# tfsf
	ez[src1] += 0.5 * tfsf_hy1[tstep]
	ez[src2] -= 0.5 * tfsf_hy2[tstep]

	# pml
	#ez[-npml-1:-1] = pml_ca[:] * ez[-npml-1:-1] + pml_cb[:] * (hy[-npml:] - hy[-npml-1:-1])


	hy[1:] -= 0.5 * (- ez[1:] + ez[:-1])
	#hy[1:-npml] -= 0.5 * (- ez[1:-npml] + ez[:-npml-1])

	# tfsf
	hy[src1] -= 0.5 * tfsf_ez1[tstep]
	hy[src2] += 0.5 * tfsf_ez2[tstep]

	# pml
	hy[-npml:] = pml_ca[:] * hy[-npml:] - pml_cb[:] * (- ez[-npml:] + ez[-npml-1:-1]) 

	if tstep%tgap == 0:
		print "tstep= %d\r" % (tstep),
		sys.stdout.flush()
		line.set_ydata(ez[:])
		plt.draw()

print ''
'''
