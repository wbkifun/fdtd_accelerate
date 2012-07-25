#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : dielectric.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 6. 23

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the class for the matter of dielectric.

===============================================================================
"""

from matter_base import *
from core.non_dispersive_core import update_non_dispersive_2update_1base
from core.non_dispersive_core import update_non_dispersive_1update_2base

class Dielectric(MatterBase):
	"""
	The dielectric matter class
	"""
	def allocate_arrays(self):
		self.efield = alloc_numpy_arrays(self.e_number_cells, 'xy')
		self.hfield = alloc_numpy_arrays(self.h_number_cells, 'z')
		
		tmp_ceb = self.dt/(ep0*self.ds)*self.unit_factor
		self.ceb = alloc_numpy_arrays(self.e_number_cells, 'xy', tmp_ceb)
		
		tmp_chb = self.dt/(mu0*self.ds)/self.unit_factor
		self.chb = alloc_numpy_arrays((1,1), 'z', tmp_chb)
		
		self.epr = alloc_numpy_arrays(self.e_number_cells, 'xy', 1)
		
		if self.grid_opt == 'efaced':
			self.matter_parameter_area_arrays = self.epr
		elif self.grid_opt == 'hfaced':
			self.matter_parameter_line_arrays = self.epr
			
			
	def set_coefficients(self):
		self.ceb = calc_with_list(self.ceb, '/', self.epr)
		
		free_numpy_arrays(self.epr)
		
		
	def update_e(self):
		if self.grid_opt == 'efaced':
			in_out_field = "out_field"
		elif self.grid_opt == 'hfaced':
			in_out_field = "in_field"
		update_non_dispersive_2update_1base(
				self.grid_opt,
				in_out_field,
				self.number_cells,
				self.efield, # update_field 
				self.hfield, # base_field
				self.ceb)
		
		
	def update_h(self):
		if self.grid_opt == 'efaced':
			in_out_field = 'in_field'
		elif self.grid_opt == 'hfaced':
			in_out_field = 'out_field'
		update_non_dispersive_1update_2base(
				self.grid_opt,
				in_out_field,
				self.number_cells,
				self.hfield, # update_field
				self.efield, # base_field
				self.chb)
		
		
#==============================================================================
# test code
#==============================================================================
if __name__ == '__main__':        
	from time import *
	from pylab import *
	from scipy import sin,pi,exp
	
	ds = 10e-9
	number_cells = [200, 200]
	grid_opt = 'efaced'
	unit_opt = 'Enorm'
	
	# construct the matter object
	space = Dielectric(ds, number_cells, grid_opt, unit_opt, 'Hz')
	space.allocate_arrays()
	Nx, Ny = number_cells[0], number_cells[1]
	Ex, Ey = space.efield[0], space.efield[1]
	Hz = space.hfield[2]
	space.set_coefficients()
	
	# for graphics using matplotlib
	ion()
	#figure(figsize=(10,5))
	figure()
	
	# for sin source
	wavelength = 300e-9
	wfreq = light_velocity*2*pi/wavelength
	
	# for data capture
	cap_t = 10 # capture_time
	cap_pt = Nx/2 # capture_point
	
	#==========================================================================
	# main time loop
	#==========================================================================
	intensity = (Hz[:,:]**2)
	imsh = imshow(intensity, cmap=cm.hot, vmin=0, vmax=0.01, origin='lower', interpolation='bilinear')
	colorbar()

	t0 = time()
	for tstep in xrange(100000):
		space.update_e()
		
		pulse = sin(wfreq*space.dt*tstep)
		space.hfield[2][Nx/3, Ny/3*2] += pulse
		
		space.update_h()
		
		if tstep/cap_t*cap_t == tstep:
			t1 = time()
			elapse_time = localtime(t1-t0-60*60*9)
			str_time = strftime('[%j]%H:%M:%S', elapse_time)
			print '%s    tstep = %d' % (str_time, tstep)
			
			intensity = (Hz[:,:]**2)
			imsh.set_array(intensity)
			#clf()
			#cla()
			draw()
