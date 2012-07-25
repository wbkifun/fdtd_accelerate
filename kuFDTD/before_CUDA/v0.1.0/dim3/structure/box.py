#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : box.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 3. 3

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the Box class.

===============================================================================
"""

from polygon import Polygon

class Box(Polygon):
    def __init__(self, matter, point1, point2):
        self.matter = matter

        x1, y1, z1 = point1[0], point1[1], point1[2]
        x2, y2, z2 = point2[0], point2[1], point2[2]
        dx, dy, dz = x1-x0, y1-y0, z1-z0
        self.polygon_points = ( \
                (x0   , y0   , z0   ), \
                (x0+dx, y0   , z0   ), \
                (x0   , y0+dy, z0   ), \
                (x0   , y0   , z0+dz), \
                (x1   , y1   , z1   ), \
                (x1-dx, y1   , z1   ), \
                (x1   , y1-dy, z1   ), \
                (x1   , y1   , z1-dz) )

        self.planes = []
