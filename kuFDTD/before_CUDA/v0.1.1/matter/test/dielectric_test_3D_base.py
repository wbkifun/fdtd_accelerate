#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : dielectric.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 6. 24

 Copyright : GNU GPL

============================== < File Description > ===========================

test code

===============================================================================
"""

from time import *
from pylab import *
from scipy import sin,pi,exp

from kufdtd.kufdtd_base import *
from kufdtd.matter.dielectric import Dielectric

ds = 10e-9
number_cells = [200, 200, 30]
grid_opt = 'efaced'
unit_opt = 'Enorm'

# construct the matter object
space = Dielectric('3D', ds, number_cells, grid_opt, unit_opt)
space.allocate_arrays()
Nx, Ny, Nz = number_cells[0], number_cells[1], number_cells[2]
Ex, Ey, Ez = space.efield[0], space.efield[1], space.efield[2]
space.set_coefficients()

# for graphics using matplotlib
ion()
#figure(figsize=(10,5))
figure()

intensity = (Ex[:,:,14]**2 + Ey[:,:,14] + Ez[:,:,14]**2)
imsh = imshow(
		intensity,
		cmap=cm.hot,
		vmin=0, vmax=0.001,
		origin='lower',
		interpolation='bilinear')
colorbar()

# for sin source
wavelength = 300e-9
wfreq = light_velocity*2*pi/wavelength

# for data capture
cap_t = 10 # capture_time
cap_pt = Nx/2 # capture_point

#--------------------------------------------------------------------------
# main time loop
#--------------------------------------------------------------------------
t0 = time()
for tstep in xrange(100000):
	space.update_e()
	
	pulse = sin(wfreq*space.dt*tstep)
	space.efield[2][Nx/3, Ny/3*2, :] += pulse
	
	space.update_h()
	
	if tstep/cap_t*cap_t == tstep:
		t1 = time()
		elapse_time = localtime(t1-t0-60*60*9)
		str_time = strftime('[%j]%H:%M:%S', elapse_time)
		print '%s    tstep = %d' % (str_time, tstep)
		
		#clf
		intensity = (Ex[:,:,14]**2 + Ey[:,:,14] + Ez[:,:,14]**2)
		imsh.set_array(intensity)
		draw()
