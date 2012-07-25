#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : dielectric.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 6. 24

 Copyright : GNU LGPL

============================== < File Description > ===========================

Define the class for the matter of dielectric.

===============================================================================
"""

from matter_base import *
from core.non_dispersive_core import update_non_dispersive_3d
from core.non_dispersive_core import update_non_dispersive_2update_1base
from core.non_dispersive_core import update_non_dispersive_1update_2base

class Dielectric(MatterBase):
	"""
	The dielectric matter class
	"""
	def allocate_arrays(self):
		ceb0 = self.dt/(ep0*self.ds)*self.unit_factor
		chb0 = self.dt/(mu0*self.ds)/self.unit_factor
		
		if self.dimension == '3D':
			self.efield = alloc_numpy_arrays(self.e_number_cells, 'xyz')
			self.hfield = alloc_numpy_arrays(self.h_number_cells, 'xyz')
			
			self.ceb = alloc_numpy_arrays(self.e_number_cells, 'xyz', ceb0)
			self.chb = alloc_numpy_arrays((1,1,1), 'xyz', chb0)
			
			self.epr = alloc_numpy_arrays(self.e_number_cells, 'xyz', 1)
			
		elif self.dimension == '2DEz':
			self.efield = alloc_numpy_arrays(self.e_number_cells, 'z')
			self.hfield = alloc_numpy_arrays(self.h_number_cells, 'xy')
			
			self.ceb = alloc_numpy_arrays(self.e_number_cells, 'z', ceb0)
			self.chb = alloc_numpy_arrays((1,1,1), 'xy', chb0)
			
			self.epr = alloc_numpy_arrays(self.e_number_cells, 'z', 1)
			
		elif self.dimension == '2DHz':
			self.efield = alloc_numpy_arrays(self.e_number_cells, 'xy')
			self.hfield = alloc_numpy_arrays(self.h_number_cells, 'z')
			
			self.ceb = alloc_numpy_arrays(self.e_number_cells, 'xy', ceb0)
			self.chb = alloc_numpy_arrays((1,1,1), 'z', chb0)
			
			self.epr = alloc_numpy_arrays(self.e_number_cells, 'xy', 1)

		if self.grid_opt == 'efaced':
			self.matter_area_arrays = self.epr
		elif self.grid_opt == 'hfaced':
			self.matter_line_arrays = self.epr

		#print 'len(epr)', len(self.epr)
		#for i in xrange(3):
		#	print 'epr[i].shape', self.epr[i].shape
			
	
	def set_coefficients(self):
		self.ceb = calc_with_list(self.ceb, '/', self.epr)
		
		#free_numpy_arrays(self.epr)
		
		
	def update_e(self):
		if self.grid_opt == 'efaced':
			in_out_field = "out_field"
		elif self.grid_opt == 'hfaced':
			in_out_field = "in_field"
			
		if self.dimension == '3D':
			update_non_dispersive_3d(
					self.grid_opt,
					in_out_field,
					self.number_cells,
					self.efield, # update_field 
					self.hfield, # base_field
					self.ceb)
			
		elif self.dimension == '2DEz':
			update_non_dispersive_1update_2base(
					self.grid_opt,
					in_out_field,
					self.number_cells,
					self.efield, # update_field 
					self.hfield, # base_field
					self.ceb)
			
		elif self.dimension == '2DHz':
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
			
		if self.dimension == '3D':
			update_non_dispersive_3d(
					self.grid_opt,
					in_out_field,
					self.number_cells,
					self.hfield, # update_field
					self.efield, # base_field
					self.chb)
			
		elif self.dimension == '2DEz':
			update_non_dispersive_2update_1base(
					self.grid_opt,
					in_out_field,
					self.number_cells,
					self.hfield, # update_field
					self.efield, # base_field
					self.chb)
			
		elif self.dimension == '2DHz':
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
