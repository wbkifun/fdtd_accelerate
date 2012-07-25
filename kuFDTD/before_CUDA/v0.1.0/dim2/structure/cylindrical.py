#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : cylindrical.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 02. 26. Thu

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the Cylindrical class.

===============================================================================
"""

from structure_base import *
from scipy.optimize import fsolve

class Cylindrical:
    def __init__(self, origin1, ellipse_coefficients1, rotation_angles1,\
            origin2, ellipse_coefficients2, rotation_angles2):
        self.origin1 = origin1
        self.origin2 = origin2
        self.a1 = ellipse_coefficients1[0]
        self.b1 = ellipse_coefficients1[1]
        self.a2 = ellipse_coefficients2[0]
        self.b2 = ellipse_coefficients2[1]
        self.angles1 = rotation_angles1
        self.angles2 = rotation_angles2


    def calc_ellipse_point(self, a, b, t, sign):
        x = t
        y = sign*b*sqrt( 1-(x/a)**2 )
        return (x, y, 0)


    def side_func1(self, x, e_point1, e_point2, rc3, axis):
        return x \
                - (1 + rc3)*(rotate(e_point1, self.angles1)[axis] + \
                        self.origin1[axis]) \
                + rc3*(rotate(e_point2, self.angles2)[axis] + \
                        self.origin2[axis]) 


    def side_func2(self, e_point1, e_point2, rc1, rc2, axis):
        return rotate(e_point1, self.angles1)[axis] \
                -rc1*rotate(e_point2, self.angles2)[axis] \
                -rc2*(self.origin2[axis] - self.origin1[axis])


    def side_main_func(self, x, g1, g2, axis, sign):
        x, t1, t2, rc1, rc2, rc3 = x[0], x[1], x[2], x[3], x[4], x[5] # ratio_coefficients

        e_point1 = self.calc_ellipse_point(self.a1, self.b1, t1, sign)
        e_point2 = self.calc_ellipse_point(self.a2, self.b2, t2, sign)

        out = [ self.side_func1(x, e_point1, e_point2, rc3, axis[0]) ]
        out.append( self.side_func1(g1, e_point1, e_point2, rc3, axis[1]) )
        out.append( self.side_func1(g2, e_point1, e_point2, rc3, axis[2]) )

        out.append( self.side_func2(e_point1, e_point2, rc1, rc2, axis[0]) )
        out.append( self.side_func2(e_point1, e_point2, rc1, rc2, axis[1]) )
        out.append( self.side_func2(e_point1, e_point2, rc1, rc2, axis[2]) )

        return out


    def find_side_intersection_points(self, direction, g1, g2):
        if direction == 'x':
            axis = (0, 1, 2)
        elif direction == 'y':
            axis = (1, 0, 2)
        elif direction == 'z':
            axis = (2, 0, 1)

        out1 = fsolve(self.side_main_func, x0=[0, 0, 0, 0, 0, 0], \
                args=(g1, g2, axis, +1), full_output=1)

        out2 = fsolve(self.side_main_func, x0=[0, 0, 0, 0, 0, 0], \
                args=(g1, g2, axis, -1), full_output=1)

        if out1[2] != 1 or out2[2] != 1:
            return None
        else:
            if out1[0][3] < 0 or out2[0][3] < 0:
                print 'Error[Cylindrical]: Ratio coefficient alpha is negative!!'
            else:
                return ( min(out1[0][0], out2[0][0]), \
                        max(out1[0][0], out2[0][0]) )


if __name__ == '__main__':
    cylinder1 = Cylindrical(
            origin1 = (0,5,7),
            ellipse_coefficients1 = (2, 2),
            rotation_angles1 = (0, 0, 0),
            origin2 = (0,5,0),
            ellipse_coefficients2 = (2, 1),
            rotation_angles2 = (0, 0, 0) )

    print cylinder1.find_side_intersection_points('y', 0, 3) 
