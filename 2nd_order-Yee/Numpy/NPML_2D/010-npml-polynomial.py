#!/usr/bin/env python

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import h5py as h5


class Fdtd2dEz:
	def __init__(s, nx, ny, dx=1):
		s.nx = nx
		s.ny = ny
		s.dx = dx
		s.dt = 0.5*s.dx		# Courant factor S=0.5

		s.ez = np.zeros((nx, ny), dtype=np.float32)
		s.hx = np.zeros_like(s.ez)
		s.hy = np.zeros_like(s.ez)
		s.cez = np.ones_like(s.ez)*0.5


	def update_h(s):
		s.hx[:,1:] -= 0.5*(s.ez[:,1:] - s.ez[:,:-1])
		s.hy[1:,:] += 0.5*(s.ez[1:,:] - s.ez[:-1,:])


	def update_e(s):
		s.ez[:-1,:-1] += s.cez[:-1,:-1]*(s.hy[1:,:-1] - s.hy[:-1,:-1] - s.hx[:-1,1:] + s.hx[:-1,:-1])


	def set_src_pt(s, x0, y0):
		s.x0 = x0
		s.y0 = y0


	def update_src_e(s, tstep):
		s.ez[s.x0, s.y0] += np.exp( -0.5*( float(tstep-100)/20 )**2 )
		#s.ez[s.x0, s.y0] += np.sin(0.5*tstep)

	

class Npml2d:
	def __init__(s, fdtd, npml, sigma_max_e, sigma_max_h, m_e, m_h, direction):
		s.npml = npml
		s.dt = fdtd.dt

		s.args_h = []
		s.args_e = []

		for axis, pm in direction:
			if axis == 0:
				psi = np.zeros((npml+1, ny), dtype=np.float32)
			elif axis == 1:
				psi = np.zeros((nx, npml+1), dtype=np.float32)


			if '+' in pm:
				ph = psi.copy()
				pe = psi.copy()
				pca_h, pcb_h = s.get_coeff(sigma_max_h, m_h, '+')
				pca_e, pcb_e = s.get_coeff(sigma_max_e, m_e, '+')
				sidx0 = slice(-npml0, None)

				s.args_h.append( (fdtd.hy, fdtd.ez, ph, pe, pcb_h, pca_e, pcb_e, sidx0, slice(-npml, None)) )
				s.args_e.append( (fdtd.hy, fdtd.ez, fdtd.cez, ph, pe, pca_h, pcb_h, pcb_e, sidx0, slice(-npml-1, -1)) )

			if '-' in pm:
				ph = psi.copy()
				pe = psi.copy()
				pca_h, pcb_h = s.get_coeff(sigma_max_h, m_h, '-')
				pca_e, pcb_e = s.get_coeff(sigma_max_e, m_e, '-')
				sidx0 = slice(None, npml+1)

				s.args_h.append( (fdtd.hy, fdtd.ez, ph, pe, pcb_h, pca_e, pcb_e, sidx0, slice(1, npml+1)) )
				s.args_e.append( (fdtd.hy, fdtd.ez, fdtd.cez, ph, pe, pca_h, pcb_h, pcb_e, sidx0, slice(None, npml)) )


	def get_coeff(s, sigma_max, m_order, direction, shift=0):
		bins = np.arange(1, s.npml+2, dtype=np.float32) + shift
		polyn = (bins / (s.npml+1)) ** m_order
		if direction == '+':
			sigma = polyn[:] * sigma_max
		elif direction == '-':
			sigma = polyn[::-1] * sigma_max

		#print('sigma', sigma)
		#plt.ion()
		#plt.plot(sigma)
		#plt.show()
		pca = (2. - sigma[:]*s.dt) / (2. + sigma[:]*s.dt)
		pcb = sigma[:]*s.dt / (2. + sigma[:]*s.dt)

		return pca, pcb


	def update_h(s):
		for h, e, ph, pe, pcb_h, pca_e, pcb_e, sidx0, sidx1 in s.args_h:
			h[sidx1] += 0.5 * (pe[1:] - pe[:-1])
			pe[:] = pca_e * pe - pcb_e * e[sidx0]
			ph[:] -= pcb_h * h[sidx0]


	def update_e(s):
		for h, e, ce, ph, pe, pca_h, pcb_h, pcb_e, sidx0, sidx1 in s.args_e:
			e[sidx1] += ce[sidx1] * (ph[1:] - ph[:-1])
			ph[:] = pca_h * ph - pcb_h * h[sidx0]
			pe[:] -= pcb_e * e[sidx0]



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


	def update2end_no_pml(s, fdtd, tmax):
		for tstep in xrange(1, tmax+1):
			fdtd.update_h()
			fdtd.update_e()
			fdtd.update_src_e(tstep)


	def update2end_pml(s, fdtd, pml, tmax):
		for tstep in xrange(1, tmax+1):
			fdtd.update_h()
			pml.update_h()
			fdtd.update_e()
			fdtd.update_src_e(tstep)
			pml.update_e()


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
			s.update2end_no_pml(fdtd, s.tmax)
			ez_open = fdtd.ez[vidx]

			# no pml
			fdtd = Fdtd1d(s.nx)
			fdtd.set_src_pt(s.src_pt)
			s.update2end_no_pml(fdtd, s.tmax)
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


	def get_ez_pml(s, npml, sigma_e, sigma_h, m_e, m_h):
		fdtd = Fdtd1d(s.nx)
		fdtd.set_src_pt(s.src_pt)
		pml = Npml1d(fdtd, npml, sigma_e, sigma_h, m_e, m_h, s.direction)
		s.update2end_pml(fdtd, pml, s.tmax)

		return fdtd.ez


	def get_max_r(s, xarray, npml, wl0, wl1, plot_on=False):
		sigma_e, sigma_h, m_e, m_h = xarray
		ez_pml = s.get_ez_pml(npml, sigma_e, sigma_h, m_e, m_h)
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



def animate_update_pml(nx, ny, src_pt, npml, sigma_e, sigma_h, m_e, m_h, direction, tmax, tgap):
	fdtd = Fdtd2dEz(nx, ny)
	fdtd.set_src_pt(src_pt[0], src_pt[1])
	pml = Npml1d(fdtd, npml, sigma_e, sigma_h, m_e, m_h, direction)

	plt.ion()
	fig = plt.figure(figsize=(12,6))
	ax = fig.add_subplot(1,1,1)
	img = ax.imshow(np.ones((nx,ny),'f').T, origin='lower', vmin=-0.01, vmax=0.01)
	ax.set_xlim(nx/2*3, nx-1)
	ax.set_ylim(ny/2*3, ny-1)

	'''
	from matplotlib.patches import Rectangle
	if '+' in direction:
		rect = Rectangle((nx-npml, 0), npml, ny, alpha=0.1)
		plt.gca().add_patch(rect)
	'''

	for tstep in xrange(1, tmax+1):
		fdtd.update_h()
		pml.update_h()
		fdtd.update_e()
		fdtd.update_src_e(tstep)
		pml.update_e()

		if tstep%tgap == 0:
			print "%d/%d(%d %%)\r" % (tstep, tmax, float(tstep)/tmax*100), 
			sys.stdout.flush()
			img.set_array(fdtd.ez.T)
			plt.draw()




if __name__ == '__main__':
	nx, ny = 1024, 1024

	src_pt = 1024 - 128
	animate_update_pml(nx, ny, (src_pt, src_pt), npml=64, sigma_e=0.0498, sigma_h=0.0469, m_e=0, m_h=0, direction=('+','',''), tmax=500, tgap=5)
	#rft = Reflectance(nx, ny, src_pt_x, src_pt_y, '+')
	#print rft.get_max_r(np.array([0.0498, 0.0469, 0, 0]), 64, 25, None, plot_on=True)
	#print rft.get_max_r(np.array([0.6168, 0.5046, 0.5745, 0.5348]), 64, 25, None, plot_on=True)

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
	results = brute(rft.get_max_r, ((0,5),(0,5),(0,5),(0,5)), args=(npml, 25, None))
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
