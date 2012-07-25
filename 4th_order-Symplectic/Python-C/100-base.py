#!/usr/bin/env python

import numpy as np
import dielectric


if __name__ == '__main__':
	nx, ny, nz = 256, 256, 240
	#nx, ny, nz = 32, 128, 128

	S = 0.743	# Courant stability factor
	# symplectic integrator coefficients
	c1, c2, c3 = 0.17399689146541, -0.12038504121430, 0.89277629949778
	d1, d2 = 0.62337932451322, -0.12337932451322
	cl = np.array([c1,c2,c3,c2,c1], dtype=np.float32)
	dl = np.array([d1,d2,d2,d1,0], dtype=np.float32)

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*12
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), 'f', order='F')
	#f = np.random.randn(nx*ny*nz).astype(np.float32).reshape((nx,ny,nz), order='F')
	cf = np.ones_like(f)*(S/24)

	ex = f.copy('A')
	ey = f.copy('A')
	ez = f.copy('A')
	hx = f.copy('A')
	hy = f.copy('A')
	hz = f.copy('A')

	cex = cf.copy('A')
	cey = cf.copy('A')
	cez = cf.copy('A')
	chx = cf.copy('A')
	chy = cf.copy('A')
	chz = cf.copy('A')

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.01)
	colorbar()

	# measure execution time
	from datetime import datetime
	t1 = datetime.now()

	# main loop
	for tn in xrange(1, 100+1):
		for sn in xrange(4):
			dielectric.update_h(
					nx, ny, nz, cl[sn],
					ex, ey, ez, hx, hy, hz, 
					chx, chy, chz)

			dielectric.update_e(
					nx, ny, nz, dl[sn],
					ex, ey, ez, hx, hy, hz, 
					cex, cey, cez)

			ex[:,ny/2,nz/2] += np.sin(0.1*(tn+sn/5.))

		dielectric.update_h(
				nx, ny, nz, cl[sn],
				ex, ey, ez, hx, hy, hz, 
				chx, chy, chz)

		'''
		if tn%10 == 0:
		#if tn == 100:
			print 'tn =', tn
			imsh.set_array( ex[nx/2,:,:]**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tn) 
		'''

	print datetime.now() - t1

	imsh.set_array( ex[nx/2,:,:]**2 )
	savefig('./%.5d.png' % tn) 
