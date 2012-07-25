#!/usr/bin/env python

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import h5py as h5


class Fdtd1d:
	def __init__(s, nx, dx=1):
		s.nx = nx
		s.dx = dx
		s.dt = 0.5*s.dx		# Courant factor S=0.5

		s.ez = np.zeros(nx, dtype=np.float32)
		s.hy = np.zeros_like(s.ez)
		s.cez = np.ones_like(s.ez)*0.5


	def update_h(s):
		s.hy[1:] += 0.5*(s.ez[1:] - s.ez[:-1])


	def update_e(s):
		s.ez[:-1] += s.cez[:-1]*(s.hy[1:] - s.hy[:-1])


	def set_src_pt(s, x0):
		s.x0 = x0


	def update_src_e(s, tstep):
		s.ez[s.x0] += np.exp( -0.5*( float(tstep-100)/20 )**2 )

	
	def set_npml(s, npml, sigma_e, sigma_h, direction):
		s.npml = npml
		s.direction = direction
		if s.direction in ['+', '-']:
			s.psi_ez = np.zeros(s.npml+1, dtype=np.float32)
			s.psi_hy = np.zeros_like(s.psi_ez)

		elif s.direction in ['+-', '-+']: 
			s.psi_ez = np.zeros(2*(s.npml+1), dtype=np.float32)
			s.psi_hy = np.zeros_like(s.psi_ez)

		else:
			print('%s is wrong direction option' % direction)
			sys.exit()

		s.pca_e = (2. - sigma_e*s.dt) / (2. + sigma_e*s.dt)
		s.pcb_e = sigma_e*s.dt / (2. + sigma_e*s.dt)
		s.pca_h = (2. - sigma_h*s.dt) / (2. + sigma_h*s.dt)
		s.pcb_h = sigma_h*s.dt / (2. + sigma_h*s.dt)


	def update_npml_h(s):
		nm = s.npml
		if '+' in s.direction:
			s.hy[-nm:] += 0.5*(s.psi_ez[-nm:] - s.psi_ez[-nm-1:-1])
			s.psi_hy[-nm-1:] -= s.pcb_h*s.hy[-nm-1:]
			s.psi_ez[-nm-1:] = s.pca_e*s.psi_ez[-nm-1:] - s.pcb_e*s.ez[-nm-1:]

		if '-' in s.direction:
			s.hy[1:nm+1] += 0.5*(s.psi_ez[1:nm+1] - s.psi_ez[:nm])
			s.psi_hy[:nm+1] -= s.pcb_h*s.hy[:nm+1]
			s.psi_ez[:nm+1] = s.pca_e*s.psi_ez[:nm+1] - s.pcb_e*s.ez[:nm+1]


	def update_npml_e(s):
		nm = s.npml
		if '+' in s.direction:
			s.ez[-nm-1:-1] += s.cez[-nm-1:-1]*(s.psi_hy[-nm:] - s.psi_hy[-nm-1:-1])
			s.psi_ez[-nm-1:] -= s.pcb_e*s.ez[-nm-1:]
			s.psi_hy[-nm-1:] = s.pca_h*s.psi_hy[-nm-1:] - s.pcb_h*s.hy[-nm-1:]

		if '-' in s.direction:
			s.ez[:nm] += s.cez[:nm]*(s.psi_hy[1:nm+1] - s.psi_hy[:nm])
			s.psi_ez[:nm+1] -= s.pcb_e*s.ez[:nm+1]
			s.psi_hy[:nm+1] = s.pca_h*s.psi_hy[:nm+1] - s.pcb_h*s.hy[:nm+1]


	def update2end_no_pml(s, tmax):
		for tstep in xrange(1, tmax+1):
			s.update_h()
			s.update_e()
			s.update_src_e(tstep)


	def update2end_pml(s, tmax):
		for tstep in xrange(1, tmax+1):
			s.update_h()
			s.update_npml_h()
			s.update_e()
			s.update_src_e(tstep)
			s.update_npml_e()


	def animate_update_pml(s, tmax, tgap):
		plt.ion()
		fig = plt.figure(figsize=(12,6))
		ax = fig.add_subplot(1,1,1)
		line, = ax.plot(s.ez)
		ax.set_ylim(-1.2, 1.2)
		ax.set_xlim(0, s.nx-1)

		from matplotlib.patches import Rectangle
		if '+' in s.direction:
			rect = Rectangle((s.nx-s.npml, -1.2), s.nx, 2.4, alpha=0.1)
			plt.gca().add_patch(rect)

		elif '-' in s.direction:
			rect = Rectangle((0, -1.2), s.npml, 2.4, alpha=0.1)
			plt.gca().add_patch(rect)

		for tstep in xrange(1, tmax+1):
			s.update_h()
			s.update_npml_h()
			s.update_e()
			s.update_src_e(tstep)
			s.update_npml_e()

			if tstep%tgap == 0:
				print "%d/%d(%d %%)\r" % (tstep, tmax, float(tstep)/tmax*100), 
				sys.stdout.flush()
				line.set_ydata(s.ez)
				line.recache()
				plt.draw()



class Reflectance:
	def __init__(s, nx, src_pt, direction):
		s.nx = nx
		s.src_pt = src_pt
		s.direction = direction
		s.tmax = s.nx * 2

		# reference
		s.fpath = '1d_%d_%d_%s.h5' % (s.nx, s.src_pt, s.direction)
		s.ez_open, s.ez_ref = s.get_ez_ref()
		s.kez_ref = np.abs( np.fft.fft(s.ez_ref) )
		s.bin_wl = 1./np.fft.fftfreq(s.nx, 1)


	def get_ez_ref(s):
		try:
			f = h5.File(s.fpath, 'r')
			ez_open = f['ez_open'].value
			ez_ref = f['ez_ref'].value

		except IOError:
			# open
			fdtd = Fdtd1d(s.nx*1.5)

			if s.direction == '+':
				vidx = slice(None, s.nx)
				src_pt = s.src_pt

			elif s.direction == '-':
				vidx = slice(-s.nx, None)
				src_pt = -s.nx + s.src_pt

			fdtd.set_src_pt(src_pt)
			fdtd.update2end_no_pml(s.tmax)
			ez_open = fdtd.ez[vidx]

			# no pml
			fdtd = Fdtd1d(s.nx)
			fdtd.set_src_pt(s.src_pt)
			fdtd.update2end_no_pml(s.tmax)
			ez_nopml = fdtd.ez
			ez_ref = ez_nopml[:] - ez_open[:]

			# save to h5
			f = h5.File(s.fpath, 'w')
			f.attrs['tmax'] = s.tmax
			f.create_dataset('ez_open', data=ez_open)
			f.create_dataset('ez_nopml', data=ez_nopml)
			f.create_dataset('ez_ref', data=ez_ref)
			f.close()

		return (ez_open, ez_ref)


	def plot_ez_ref(s):
		f = h5.File(s.fpath, 'r')
		ez_open = f['ez_open'].value
		ez_nopml = f['ez_nopml'].value
		ez_ref = f['ez_ref'].value
		f.close()

		plt.ioff()
		fig = plt.figure()
		ax1 = fig.add_subplot(3,1,1)
		ax2 = fig.add_subplot(3,1,2)
		ax3 = fig.add_subplot(3,1,3)
		ax1.plot(ez_open)
		ax2.plot(ez_nopml)
		ax3.plot(ez_ref)
		ax1.set_xticklabels([])
		ax2.set_xticklabels([])
		ax1.set_ylim(-1.2, 1.2)
		ax2.set_ylim(-1.2, 1.2)
		ax3.set_ylim(-1.2, 1.2)
		ax1.set_xlim(0, s.nx-1)
		ax2.set_xlim(0, s.nx-1)
		ax3.set_xlim(0, s.nx-1)
		#plt.show()
		plt.savefig('./ez_ref_%s.png' % s.direction)


	def get_ez_pml(s, npml, sigma_e, sigma_h):
		fdtd = Fdtd1d(s.nx)
		fdtd.set_src_pt(s.src_pt)
		fdtd.set_npml(npml, sigma_e, sigma_h, s.direction)
		fdtd.update2end_pml(s.tmax)

		return fdtd.ez


	def get_max_r(s, xarray, npml, wl0, wl1, plot_on=False):
		sigma_e, sigma_h = xarray
		ez_pml = s.get_ez_pml(npml, sigma_e, sigma_h)
		ez = ez_pml[:] - s.ez_open[:]
		kez = np.abs( np.fft.fft(ez) )
		reflectance = (kez/s.kez_ref)**2 * 100

		if wl1 == None: sbool = (s.bin_wl>=wl0)
		else: sbool = (s.bin_wl>=wl0) * (s.bin_wl<=wl1)
		max_r = reflectance[sbool].max()

		if plot_on:
			plt.ioff()
			fig = plt.figure()
			ax1 = fig.add_subplot(2,1,1)
			ax2 = fig.add_subplot(2,1,2)
			ax1.plot(ez_pml)
			ax2.plot(ez)
			ax1.set_xticklabels([])
			ax1.set_ylim(-1.2, 1.2)
			ax2.set_ylim(-1.2, 1.2)
			ax1.set_xlim(0, s.nx-1)
			ax2.set_xlim(0, s.nx-1)
			plt.savefig('./ez_pml_%s_%dcell_%g_%g.png' % (s.direction, npml, sigma_e, sigma_h))

			fig.clf()
			ax = fig.add_subplot(1,1,1)
			ax.plot(s.bin_wl[sbool], kez[sbool], 'o-')
			ax.set_xlim(0, -min(s.bin_wl))
			ax.set_xlabel('Wavelength (dx)')
			ax.set_ylabel('Amplitude (A.U.)')
			plt.savefig('./kspace_ez_%s_%dcell_%ddx_%g_%g.png' % (s.direction, npml, wl0, sigma_e, sigma_h))

			fig.clf()
			ax = fig.add_subplot(1,1,1)
			ax.plot(s.bin_wl[sbool], reflectance[sbool], 'o-')
			if max_r >= 100:
				ax.set_ylim(-0, 0.2)
			ax.set_xlim(0, -min(s.bin_wl))
			ax.set_xlabel('Wavelength (dx)')
			ax.set_ylabel('Reflectance (%)')
			plt.savefig('./reflectance_%s_%dcell_%ddx_%g_%g.png' % (s.direction, npml, wl0, sigma_e, sigma_h))

		#print(sigma_e, sigma_h, max_r)
		return max_r



if __name__ == '__main__':
	nx = 1024

	src_pt = 1024 - 128
	rft = Reflectance(nx, src_pt, '+')
	'''
	rft.plot_ez_ref()
	rft.get_max_r(np.array([1509.0, 1.0014]), 16, 1, None, plot_on=True)
	rft.get_max_r(np.array([1509.0, 1.0014]), 16, 25, None, plot_on=True)
	rft.get_max_r(np.array([60, 1]), 16, 1, None, plot_on=True)
	print rft.get_max_r(np.array([60, 1]), 16, 25, None, plot_on=True)
	print rft.get_max_r(np.array([60, 1]), 8, 25, None, plot_on=True)
	print rft.get_max_r(np.array([60, 1]), 4, 25, None, plot_on=True)
	print rft.get_max_r(np.array([60, 1]), 2, 25, None, plot_on=True)
	print rft.get_max_r(np.array([60, 1]), 1, 25, None, plot_on=True)
	print rft.get_max_r(np.array([0.2, 0.2]), 16, 25, None, plot_on=True)
	print rft.get_max_r(np.array([0.0498, 0.0469]), 64, 25, None, plot_on=True)
	print rft.get_max_r(np.array([0.0947, 0.0861]), 32, 25, None, plot_on=True)
	print rft.get_max_r(np.array([0.1866, 0.1550]), 16, 25, None, plot_on=True)
	print rft.get_max_r(np.array([0.2954, 0.2305]), 10, 25, None, plot_on=True)
	print rft.get_max_r(np.array([0.2464, 0.2132]), 8, 25, None, plot_on=True)
	'''
	fdtd = Fdtd1d(nx)
	fdtd.set_src_pt(src_pt)
	fdtd.set_npml(64, 0.0498, 0.0469, '+')
	fdtd.animate_update_pml(500,5)

	'''
	src_pt = 128
	rft = Reflectance(nx, src_pt, '-')
	rft.plot_ez_ref()
	rft.get_max_r(np.array([1, 60]), 16, 1, None, plot_on=True)
	rft.get_max_r(np.array([1, 60]), 16, 25, None, plot_on=True)
	'''

	# Optimization using the Simulated annealing
	'''
	from scipy.optimize import anneal
	from datetime import datetime
	t0 = datetime.now()
	npml = int(sys.argv[1])
	for sa in ['fast']:#, 'cauchy', 'boltzmann']:
		results = anneal(rft.get_max_r, np.array([0,0]), args=(npml, 25, None), schedule=sa, full_output=True)
		print('[%s][%s] npml = %d, %s' % (datetime.now() - t0, sa, npml, results))
	'''

	# Optimization using the Brute force at the range [0,1]
	'''
	from scipy.optimize import brute
	from datetime import datetime
	t0 = datetime.now()
	npml = int(sys.argv[1])
	#results = brute(rft.get_max_r, ((0,1),(0,1)), args=(npml, 25, None), full_output=True, finish=None)
	results = brute(rft.get_max_r, ((0,1),(0,1)), args=(npml, 25, None))
	print('[%s] npml = %d, %s' % (datetime.now() - t0, npml, results))
	'''

	# Scan the sigmas
	'''
	npml = int(sys.argv[1])
	rr = np.zeros((100,100), dtype=np.float32)
	for i, sigma_e in enumerate( np.arange(0, 1, 0.01) ):
		for j, sigma_h in enumerate( np.arange(0, 1, 0.01) ):
			rr[i,j] = rft.get_max_r(np.array([sigma_e, sigma_h]), npml, 25, None)
			print "(%g,%g)\r" % (sigma_e, sigma_h),
			sys.stdout.flush()

	np.save('./scan_sigma_%dcell_25dx_0.1_50_5.npy' % npml, rr)
	'''
