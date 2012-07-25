#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : matter_base.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 01. 31. Thu

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the parent class of the matter classes.

===============================================================================
"""

from kufdtd.kufdtd_base import *
#print 'light_velocity=%s' % light_velocity

class MatterBase:
	"""
	The parent class of the matter classes
	"""
	def __init__(self, ds, number_cells, grid_opt, unit_opt, dim2_opt):
		self.ds = ds
		self.dt = ds/(2*light_velocity)
		self.number_cells = number_cells
		self.grid_opt = grid_opt
		self.unit_opt = unit_opt
		self.dim2_opt = dim2_opt
		
		self.unit_factor = set_unit(unit_opt)
		
		if dim2_opt == 'Ez':
			if grid_opt == 'efaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([2])
				self.h_number_cells = sc.array(number_cells) + sc.array([1,1])
			elif grid_opt == 'hfaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([1])
				self.h_number_cells = sc.array(number_cells) + sc.array([2,2])
		elif dim2_opt == 'Hz':
			if grid_opt == 'efaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([2,2])
				self.h_number_cells = sc.array(number_cells) + sc.array([1])
			elif grid_opt == 'hfaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([1,1])
				self.h_number_cells = sc.array(number_cells) + sc.array([2])
				
	def __repr__(self):
		return """
			Dimension of the Simulation = 2D %s
			Cell Size(ds) = %.2e
			Infinitesimal Time(dt) = %.2e
			The number of the cells = %s
			The Yee grid option = %s
			The unit option = %s""" % (
					self.dim2_opt,
					self.ds, 
					self.dt,
					self.grid_opt,
					self.unit_opt)
