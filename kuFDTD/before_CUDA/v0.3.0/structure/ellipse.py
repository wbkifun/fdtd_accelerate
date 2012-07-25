#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : ellipse.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 6. 25

 Copyright : GNU LGPL

============================== < File Description > ===========================

Define the classes.
	Ellipse3D
	Ellipse2D

===============================================================================
"""

from structure_base import *

class Ellipse2D:
	def __init__(self):
		pass


	def calc_ellipse_point(self, a, b, t, sign):
		x = t
		y = sign*b*sqrt( 1-(x/a)**2 )
		return (x, y)


	def func(self, x):
		ro = self.rotate_origin
		point =  (x0[0]-ro[0], x0[1]-ro[1])
		
		return x - rotate(point1, self.angle)[axis] - ro[axis]


	def find_intersection_points(self, direction):
		if direction == 'x':
			axis = 
		out1 = fsolve(self.func, x0, args=(0, 0), full_output=1)


