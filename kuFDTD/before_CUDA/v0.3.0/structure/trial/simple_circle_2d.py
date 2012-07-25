#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : simple_circle_2d.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 7. 23

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


	def is_in(self, point):
		a, b, r = self.a, self.b, self.r
		x, y = point

		d = sc.sqrt( (x-a)**2 + (y-b)**2 )

		if d <= r:
			return True
		else:
			return False
