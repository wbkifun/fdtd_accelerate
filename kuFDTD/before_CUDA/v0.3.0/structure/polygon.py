#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : polygon.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 2. 26

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the Polygon class.

===============================================================================
"""

from structure_base import *
from polygonal_plane import PolygonalPlane

class Polygon:
    def __init__(self, name, matter, points):
        self.__name__ = name
        self.matter = matter
        self.polygon_points = points
        self.planes = []
        self.discretized_effective_region = (0,0,0,0,0,0)
        self.intersection_points_arrays = []


    def set_accuracy(self, ds, accuracy_ratio=0.0001):
        self.ds = ds
        self.accuracy = self.ds*accuracy_ratio


    def set_effective_region(self):
        x, y, z = [], [], []
        for point in self.polygon_points:
            x.append( point[0] )
            y.append( point[1] )
            z.append( point[2] )
        self.effective_region = (\
                min(x), max(x), min(y), max(y), min(z), max(z) )

        DER = self.discretized_effective_region = \
                ( sc.array(self.effective_region)/self.ds ).astype('i')
        DER[1::2] += 1

        DERS = self.discrete_effective_region_sides = \
                DER[1::2] - DER[::2]

        self.discrete_effective_region_center = \
                DER[::2] + 0.5*DERS[:]


    def is_points_on_line(point1, point2, point3):
        points = [point1, point2, point3]
        vector12 = sc.array(point2) - sc.array(point1)
        vector13 = sc.array(point3) - sc.array(point1)
        ratio = vector12 - vector13
        if ratio[0] == ratio[1] and ratio[0] == ratio[2]:
            return True
        else:
            return False


    def is_exist_plane(self, plane):
        result = False
        for exist_plane in self.planes:
            if plane.is_same_plane(exist_plane):
                result = True
                break

        return result


    def is_intersect_plane(self, plane):
        Npt = len(self.polygon_points)

        result = False
        for i in xrange(Npt-1):
            for j in xrange(i+1, Npt):
                point1 = self.polygon_points[i]
                point2 = self.polygon_points[j]

                if (point1 in plane.points) or (point2 in plane.points):
                    pass
                elif plane.eval(point1)*plane.eval(point2) < 0:
                    result = True
            if result:
                break

        return result


    def make_planes(self):
        Npt = len(self.polygon_points)

        for i in xrange(Npt-2):
            for j in xrange(i+1, Npt-1):
                for k in xrange(j+1, Npt):
                    point1 = self.polygon_points[i]
                    point2 = self.polygon_points[j]
                    point3 = self.polygon_points[k]

                    if !( is_points_on_line(point1, point2, point3) ):
                        plane = PolygonalPlane(\
                                point1, point2, point3)

                        if is_exist_plane( plane ):
                            pass
                        elif is_intersect_plane( plane ):
                            pass
                        else:
                            for l in xrange(k+1, Npt):
                                point4 = self.polygon_points[l]
                                if plane.eval(point4) < self.accuracy:
                                    plane.points.append( point4 )
                            plane.points = sort_points( plane.points )

                            self.planes.append( plane )


    def make_intersection_points_arrays(self):
        self.make_planes()

        DER = self.discretized_effective_region
        DER[1::2] += 1
        Nx, Ny, Nz = self.discretized_effective_region_sides
        ISPTs_arrays = []
        ISPTs_arrays.append( sc.zeros((2,Ny,Nz), 'f') )
        ISPTs_arrays.append( sc.zeros((2,Nx,Nz), 'f') )
        ISPTs_arrays.append( sc.zeros((2,Nx,Ny), 'f') )

        for axis in [x_axis, y_axis, z_axis]:
            axis_list = [x_axis, y_axis, z_axis]
            axis_list.pop(axis)
            i_axis, j_axis = axis_list
            i0, j0 = DER[i_axis*2], DER[j_axis*2]

            for i in xrange( N[i_axis] ):
                for j in xrange( N[j_axis] ):
                    ISPTs = [] # intersection points

                    for plane in self.planes:
                        x0, y0 = (i0+i)*self.ds, (j0+j)*self.ds
                        ISPT = plane.intersection_point(axis, x0, y0)/self.ds
                        if ISPT != None:
                            ISPTs.append(ISPT)

                    if len(ISPTs) > 2:
                        print "Error: the number of intersection points in \
                                structure %s is wrong. -> %d" \
                                % (self.__name__, len(ISTPs))
                    elif len(ISPTs) == 2:
                        ISPTs_arrays[axis][0][i,j] = min(ISPTs)
                        ISPTs_arrays[axis][1][i,j] = max(ISPTs)
                    elif len(ISPTs) == 1:
                        ISPTs_arrays[axis][0][i,j] = min(ISPTs)
                        ISPTs_arrays[axis][1][i,j] = min(ISPTs) + \
                                self.accuracy*10
                    elif len(ISPTs) == 0:
                        ISPTs_arrays[axis][0][i,j] = sc.nan
                        ISPTs_arrays[axis][1][i,j] = sc.nan


        self.intersection_points_arrays = ISPTs_arrays
