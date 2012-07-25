#!/usr/bin/env python

import numpy as np
import dielectric


def set_c(cf, pt):
	cf[:,:,:] = 0.5
	if pt[0] != None: cf[pt[0],:,:] = 0
	if pt[1] != None: cf[:,pt[1],:] = 0
	if pt[2] != None: cf[:,:,pt[2]] = 0

	return cf


if __name__ == '__main__':
	nx, ny, nz = 512, 512, 335

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*12
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), np.float32)

	ex_cpu = np.zeros_like(f)
	ey_cpu = np.zeros_like(f)
	ez_cpu = np.zeros_like(f)
	hx_cpu = np.zeros_like(f)
	hy_cpu = np.zeros_like(f)
	hz_cpu = np.zeros_like(f)

	cex_cpu = set_c(f,(None,-1,-1)).copy()
	cey_cpu = set_c(f,(-1,None,-1)).copy()
	cez_cpu = set_c(f,(-1,-1,None)).copy()
	chx_cpu = set_c(f,(None,0,0)).copy()
	chy_cpu = set_c(f,(0,None,0)).copy()
	chz_cpu = set_c(f,(0,0,None)).copy()

	'''
	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.001)
	colorbar()
	'''

	# measure kernel execution time
	from datetime import datetime
	t1 = datetime.now()

	# main loop
	for tn in xrange(1, 1001):
		dielectric.update_e(
				8, nx, ny, nz,
				ex_cpu, ey_cpu, ez_cpu, hx_cpu, hy_cpu, hz_cpu,
				cex_cpu, cey_cpu, cez_cpu)

		ex_cpu[:,ny/2,nz/2] += np.sin(0.1*tn)

		dielectric.update_h(
				8, nx, ny, nz,
				ex_cpu, ey_cpu, ez_cpu, hx_cpu, hy_cpu, hz_cpu,
				chx_cpu, chy_cpu, chz_cpu)

		'''
		if tn%10 == 0:
		#if tn == 100:
			print 'tn =', tn
			imsh.set_array( ex_cpu[nx/2,:,:]**2 )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
		'''

	print datetime.now() - t1
