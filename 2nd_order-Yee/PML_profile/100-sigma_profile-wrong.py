#!/usr/bin/env python

import numpy as np
from scipy.optimize import newton	# root find
from scipy.constants import c as C, mu_0 as MU0, epsilon_0 as EP0

# setup
dx = 10e-9		# m
dt = 0.5 * dx/C	# Courant factor S=0.5
wavelength = 300e-9
frequency = C / wavelength
omega = 2 * np.pi * frequency


def only_sigma_f(field0, nx, nkx, sigma_dx):
	i = np.sign(nx) * np.arange(abs(nx))
	field = field0 * np.exp(1j * nkx * dx * i) * np.exp(-sigma_dx * nkx * i / omega)
	return field.real


def only_sigma_k(field0, nx, nkx, sigma_dx):
	i = np.sign(nx) * np.arange(abs(nx))
	field = field0 * np.exp(1j * (1 + sigma_dx / 2) * nkx * dx * i) * np.exp(-sigma_dx * i)
	return field.real


def calc_nkx(epr, angle):
	angle = np.radians(angle)
	w2 = (np.sin(omega * dt / 2) * 2 / dt)**2
	kx2 = lambda nk: (np.sin(nk * np.cos(angle) * dx / 2) * 2 / dx)**2
	ky2 = lambda nk: (np.sin(nk * np.sin(angle) * dx / 2) * 2 / dx)**2
	func = lambda nk: epr * EP0 * MU0 * w2 - kx2(nk) - ky2(nk)
	nk = newton(func, 2 * np.pi / wavelength)
	return nk * np.cos(angle)


def find_optimal_sigma_dx(nx, nkx): 
	f1 = lambda sigma_dx: only_sigma_f(1, nx, nkx, sigma_dx)[nx-1]
	func = lambda sigma_dx: only_sigma_f(f1(sigma_dx), -nx, -nkx, sigma_dx)[nx-1]

	print func(1e8)

	return newton(func, 1e7, tol=0.01)



if __name__ == '__main__':
	nx = 150
	field1, field2 = [np.zeros(2 * nx) for i in range(2)]
	polynomial = (np.linspace(0,1,nx) ** 4)

	nkx = calc_nkx(epr=1, angle=0)
	print 'nkx = ', nkx
	
	sigma_dx_max1 = 5e6
	sigma_dx1 = polynomial * sigma_dx_max1
	field1[:nx] = only_sigma_f(1, nx, nkx, sigma_dx1)
	field1[nx:] = only_sigma_f(field1[nx-1], -nx, -nkx, sigma_dx1[::-1])
	print 'sigma_dx_max = ', sigma_dx_max1
	print 'only_sigma_f : %g' % field1[-1]

	sigma_dx_max2 = 0.05
	sigma_dx2 = polynomial * sigma_dx_max2
	field2[:nx] = only_sigma_k(1, nx, nkx, sigma_dx2)
	field2[nx:] = only_sigma_k(field2[nx-1], -nx, -nkx, sigma_dx2[::-1])
	print 'sigma_dx_max = ', sigma_dx_max2
	print 'only_sigma_k : %g' % field2[-1]

	# plot
	import matplotlib.pyplot as plt
	plt.ion()
	plt.subplot(3,1,1)
	plt.plot(np.concatenate((sigma_dx2, sigma_dx2[::-1])))
	plt.subplot(3,1,2)
	plt.plot(field1)
	plt.subplot(3,1,3)
	plt.plot(field2)
	plt.show()