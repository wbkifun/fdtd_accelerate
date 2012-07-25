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

	
	def set_npml(s, npml, sigma_e, sigma_h):
		s.npml = npml
		s.psi_ez = np.zeros(s.npml+1, dtype=np.float32)
		s.psi_hy = np.zeros_like(s.psi_ez)

		s.pca_e = (2. - sigma_e*s.dt) / (2. + sigma_e*s.dt)
		s.pcb_e = sigma_e*s.dt / (2. + sigma_e*s.dt)
		s.pca_h = (2. - sigma_h*s.dt) / (2. + sigma_h*s.dt)
		s.pcb_h = sigma_h*s.dt / (2. + sigma_h*s.dt)


	def update_npml_h(s):
		s.hy[1:s.npml+1] += 0.5*(s.psi_ez[1:] - s.psi_ez[:-1])
		s.psi_hy[:] -= s.pcb_h*s.hy[:s.npml+1]
		s.psi_ez[:] = s.pca_e*s.psi_ez[:] - s.pcb_e*s.ez[:s.npml+1:]


	def update_npml_e(s):
		s.ez[-s.npml-1:-1] += s.cez[-s.npml-1:-1]*(s.psi_hy[1:] - s.psi_hy[:-1])
		s.psi_ez[:] -= s.pcb_e*s.ez[-s.npml-1:]
		s.psi_hy[:] = s.pca_h*s.psi_hy[:] - s.pcb_h*s.hy[-s.npml-1:]


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
		rect = Rectangle((s.nx-s.npml, -1.2), s.npml, 2.4, alpha=0.1)
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

		if s.direction == '+':
			pass
		elif s.direction == '-':
			pass
		else:
			print('Wrong direction option.')
			sys.exit()

		# reference
		s.fpath = '1d_%d_%d.h5' % (s.nx, s.src_pt)
		s.ez_open, s.ez_ref = s.get_ez_ref()
		s.kez_ref = np.abs( np.fft.fft(s.ez_ref) )
		s.wl = 1./np.fft.fftfreq(s.nx, 1)


	def get_ez_ref(s):
		try:
			f = h5.File(s.fpath, 'r')
			ez_open = f['ez_open'].value
			ez_ref = f['ez_ref'].value

		except IOError:
			# open
			fdtd = Fdtd1d(nx*1.5)

			if s.direction == '+':
				vidx = slice(None, s.nx)
				src_pt = s.src_pt

			elif s.direction == '-':
				vidx = slice(-s.nx, None)
				src_pt = -s.src_pt

			fdtd.set_src_pt(src_pt)
			fdtd.update2end_no_pml(s.tmax)
			ez_open = fdtd.ez[vidx]

			# no pml
			fdtd = Fdtd1d(nx)
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

		plt.ion()
		fig = plt.figure()
		ax1 = fig.add_subplot(3,1,1)
		ax2 = fig.add_subplot(3,1,2)
		ax3 = fig.add_subplot(3,1,3)
		ax1.plot(ez_open)
		ax2.plot(ez_nopml)
		ax3.plot(ez_ref)
		#plt.show()
		plt.savefig('./ez_ref.png')


	def get_ez_pml(s, npml, sigma_e, sigma_h):
		fdtd = Fdtd1d(s.nx)
		fdtd.set_src_pt(s.src_pt)
		fdtd.set_npml(npml, sigma_e, sigma_h)
		fdtd.update2end_pml(s.tmax)

		return fdtd.ez


	def get_max_r(s, xarray, npml, wl0, wl1, plot_on=False):
		sigma_e, sigma_h = xarray
		ez_pml = s.get_ez_pml(npml, sigma_e, sigma_h)
		ez = ez_pml[:] - s.ez_open[:]
		kez = np.abs( np.fft.fft(ez) )
		kez_ratio = kez/s.kez_ref

		if wl1 == None: sbool = (s.wl>=wl0)
		else: sbool = (s.wl>=wl0) * (s.wl<=wl1)

		if plot_on:
			plt.ion()
			plt.plot(s.wl[sbool], kez_ratio[sbool], 'o-')
			plt.show()

		max_r = kez_ratio[sbool].max()
		#print(sigma_e, sigma_h, max_r)
		return max_r



if __name__ == '__main__':
	nx = 1024
	src_pt = 1024 - 128

	rft = Reflectance(nx, src_pt, '+')
	#print( rft.get_max_r(np.array([0.002,0.002]), 32, 20, None) )
	#print( rft.get_max_r(np.array([4545.92,1.00150]), 32, 20, None) )
	#print( rft.get_max_r(np.array([4605, 1]), 8, 1, None, True) )

	#rft.plot_ez_ref()
	'''
	fdtd = Fdtd1d(nx)
	fdtd.set_src_pt(src_pt)
	fdtd.set_npml(npml=8, sigma_e=4605, sigma_h=1)
	fdtd.animate_update_pml(1000, 50)
	'''

	n = 5000
	rs = np.zeros(n)
	se_list = np.arange(n)
	npml_list = [64, 32, 16, 10, 8]
	f = h5.File('scan_sigma_e_1.h5', 'w')
	f.attrs['bin'] = n
	for npml in npml_list:
		print('npml = %d' % npml)
		for i, sigma_e in enumerate(se_list):
			rs[i] = rft.get_max_r(np.array([sigma_e, 1]), npml, 25, None)

		f.create_dataset('%d' % npml, data=rs)
	

	'''
	from scipy.optimize import anneal
	npml = 8
	print('npml = %d' % npml)
	results = anneal(rft.get_max_r, np.array([0,0]), args=(npml, 20, None), full_output=True)
	print(results)
	'''

	'''
	from scipy.optimize import anneal
	npml_list = [64, 32, 16, 10, 8]
	for npml in npml_list:
		print('npml = %d' % npml)
		results = anneal(rft.get_max_r, np.array([0,0]), args=(npml, 20, None), full_output=True)
		print(results)
		print('')
	'''
