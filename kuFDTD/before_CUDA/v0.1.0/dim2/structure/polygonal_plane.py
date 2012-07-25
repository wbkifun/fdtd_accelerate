#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : polygonal_plane.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 3. 3. Thu

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the Plane class for polygon.

===============================================================================
"""

from structure_base import *

class PolygonalPlane:
    def __init__(self, point1, point2, point3):
        self.points = [point1, point2, point3]
        vector12 = sc.array(point2) - sc.array(point1)
        vector13 = sc.array(point3) - sc.array(point1)
        normal_vector = sc.cross(vector12, vector13)
        self.a = normal_vector[0]
        self.b = normal_vector[1]
        self.c = normal_vector[2]
        self.d = sc.dot(normal_vector, sc.array(point1))


    def eval(self, point):
        x, y, z = point
        return self.a*x + self.b*y + self.c*z - d


    def intersection_point(self, axis, x0, y0):
        if aixs == x_axis:
            z = (self.d - self.b*x0 - self.c*y0)/self.a
            point = (z, x0, y0)
        elif aixs == y_axis:
            z = (self.d - self.a*x0 - self.c*y0)/self.b
            point = (x0, z, y0)
        elif aixs == z_axis:
            z = (self.d - self.a*x0 - self.b*y0)/self.c
            point = (x0, y0, z)

        if point_in_boundary(self.points, point):
            return z
        else:
            return None


    def is_same_plane(self, plane2):
        if (plane2.points[0] in self.points) and \
                (plane2.points[1] in self.points) and \
                (plane2.points[2] in self.points):
            return True
        else:
            return False



if __name__ == '__main__':
    pass
