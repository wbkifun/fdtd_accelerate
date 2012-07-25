#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : matter_base.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 6. 24

 Copyright : GNU LGPL

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
	def __init__(self, dimension, ds, number_cells, grid_opt, unit_opt):
		self.dimension = dimension
		self.ds = ds
		self.dt = ds/(2*light_velocity)
		self.number_cells = number_cells
		self.grid_opt = grid_opt
		self.unit_opt = unit_opt
		
		self.unit_factor = set_unit(unit_opt)
		
		if dimension == '3D':
			if grid_opt == 'efaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([2,2,2])
				self.h_number_cells = sc.array(number_cells) + sc.array([1,1,1])
			elif grid_opt == 'hfaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([1,1,1])
				self.h_number_cells = sc.array(number_cells) + sc.array([2,2,2])
		elif dimension == '2DEz':
			if grid_opt == 'efaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([2])
				self.h_number_cells = sc.array(number_cells) + sc.array([1,1])
			elif grid_opt == 'hfaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([1])
				self.h_number_cells = sc.array(number_cells) + sc.array([2,2])
				
		elif dimension == '2DHz':
			if grid_opt == 'efaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([2,2])
				self.h_number_cells = sc.array(number_cells) + sc.array([1])
			elif grid_opt == 'hfaced':
				self.e_number_cells = sc.array(number_cells) + sc.array([1,1])
				self.h_number_cells = sc.array(number_cells) + sc.array([2])
				
				
	def __repr__(self):
		return """
			Dimension of the Simulation = %s
			Cell Size(ds) = %.2e
			Infinitesimal Time(dt) = %.2e
			The number of the cells = %s
			The Yee grid option = %s
			The unit option = %s""" % (
					self.demension,
					self.ds, 
					self.dt,
					self.grid_opt,
					self.unit_opt)
