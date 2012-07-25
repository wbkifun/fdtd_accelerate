#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : circle_2d.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 7. 2

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the structure classes.
	Circle2D

===============================================================================
"""

from kufdtd.structure.structure_base_2d import *

import numpy

class Circle2D:
	def __init__(self, o_pt, r, epr):
		self.a = o_pt[x_axis]
		self.b = o_pt[y_axis]
		self.r = r
		self.epr = epr


	def calc_intersection_points(self, point, grid_direction):
		a, b, r = self.a, self.b, self.r

		if grid_direction == x_axis:
			y = point

			cterm = sc.sqrt(r**2 - (y-b)**2)
			if type(cterm) is not numpy.complex128:
				x1 = a - cterm
				x2 = a + cterm

				unit_grad_vec1 = ( (x1-a)/r, (y-b)/r )
				unit_grad_vec2 = ( (x2-a)/r, (y-b)/r )
				theta1 = sc.arccos( rotate(unit_grad_vec1, -0.5*pi)[0] )
				theta2 = sc.arccos( rotate(unit_grad_vec2, 0.5*pi)[0] )

				return x1, x2, theta1, theta2

			else:
				return None


		elif grid_direction == y_axis:
			x = point

			cterm = sc.sqrt(r**2 - (x-a)**2)
			if type(cterm) is not numpy.complex128:
				#print cterm
				#print type(cterm)
				y1 = b - cterm
				y2 = b + cterm
				#print 'x=%g, y1=%g, y2=%g' % (x, y1, y2)

				unit_grad_vec1 = ( (x-a)/r, (y1-b)/r )
				unit_grad_vec2 = ( (x-a)/r, (y2-b)/r )
				theta1 = sc.arccos( rotate(unit_grad_vec1, -0.5*pi)[1] )
				theta2 = sc.arccos( rotate(unit_grad_vec2, 0.5*pi)[1] )

				return y1, y2, theta1, theta2

			else:
				return None