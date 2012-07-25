#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : ellipse.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 6. 27

 Copyright : GNU LGPL

============================== < File Description > ===========================

Define the classes.
	Circle3D
	Circle2D

===============================================================================
"""

from structure_base import *

class Circle2D:
	def __init__(self, origin, radius):
		self.origin = origin
		self.radius = radius


	def eval_x(self, y, sign):
		a, b = self.origin
		r = self.radius
		x = a + sign*sc.sqrt(r**2 - (y-b)**2)
		return x


	def eval_y(self, x, sign):
		a, b = self.origin
		r = self.radius
		y = b + sign*sc.sqrt(r**2 - (x-a)**2)
		return y


	def func(self, x):
		ro = self.rotate_origin
		point =  (x0[0]-ro[0], x0[1]-ro[1])
		
		return x - rotate(point1, self.angle)[axis] - ro[axis]


	def find_intersection_points(self, direction):
		if direction == 'x':
			axis = 
		out1 = fsolve(self.func, x0, args=(0, 0), full_output=1)


