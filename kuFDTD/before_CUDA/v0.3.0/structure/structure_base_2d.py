#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : structure_base_2d.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 6. 25

 Copyright : GNU LGPL

============================== < File Description > ===========================

Define the geometrical functions for 2-dimensional structure.

===============================================================================
"""

from kufdtd.kufdtd_base import *

def make_matrix_vector(point):
	if len(point) == 3:
		V = sc.zeros((3,1), 'f')
		V[0,0], V[1,0], V[2,0] = point[0], point[1], point[2]
	elif len(point) == 2:
		V = sc.zeros((2,1), 'f')
		V[0,0], V[1,0] = point[0], point[1]
		
	return sc.matrix(V)


def matrix_vector2point(vector):
	point = []
	if len(vector) == 3:
		point = (vector[0,0], vector[1,0], vector[2,0])
	elif len(vector) == 2:
		point = (vector[0,0], vector[1,0])
				
	return point
			
	
def rotate(point, rotation_angles):
	X = make_matrix_vector(point)
	theta = rotation_angles
	
	if len(point) == 3:
		M = sc.identity(3)
		M[1,1], M[1,2] = sc.cos(theta[0]), sc.sin(theta[0])
		M[2,1], M[2,2] = -sc.sin(theta[0]), sc.cos(theta[0])
		Rx = sc.matrix(M)
		
		M = sc.identity(3)
		M[0,0], M[0,2] = sc.cos(theta[1]), -sc.sin(theta[1])
		M[2,0], M[2,2] = sc.sin(theta[1]), sc.cos(theta[1])
		Ry = sc.matrix(M)
		
		M = sc.identity(3)
		M[0,0], M[0,1] = sc.cos(theta[2]), sc.sin(theta[2])
		M[1,0], M[1,1] = -sc.sin(theta[2]), sc.cos(theta[2])
		Rz = sc.matrix(M)
		#print '%s\n\n%s\n\n%s\n\n%s' %(X, Rx, Ry, Rz)
		
		X2 = Rz*Ry*Rx*X
		
	elif len(point) == 2:
		M = sc.identity(2)
		M[0,0], M[0,1] = sc.cos(theta), -sc.sin(theta)
		M[1,0], M[1,1] = sc.sin(theta), sc.cos(theta)
		R = sc.matrix(M)
		
		X2 = R*X 
		
	return matrix_vector2point( X2 )


def distance(pt1, pt2):
	if len(pt1) == 3:
		return sc.sqrt( (pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 + (pt2[2]-pt1[2])**2 )
	elif len(pt1) == 2:
		return sc.sqrt( (pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 )
	
	
def sort_points(points):
	if len(points) < 2:
		return points
	
	sorted_points = []
	
	p1 = points.pop(0)
	sorted_points.append(p1)
	min_cos_theta = 1
	min_index = 0
	for i, p2 in enumerate( points ):
		for p3 in points[i+1:]:
			a = distance(p1, p2)
			b = distance(p1, p3)
			c = distance(p2, p3)
			cos_theta = (a**2 + b**2 -c**2)/(2*a*b)
			if cos_theta < min_cos_theta:
				min_index = i
				min_cos_theta = cos_theta
	sorted_points.append( points.pop(min_index) )
	
	cos_theta = []
	while( len(points) > 1 ):
		for p3 in points:
			p1 = sorted_points[-1]
			p2 = sorted_points[-2]
			a = distance(p1, p2)
			b = distance(p1, p3)
			c = distance(p2, p3)
			cos_theta.append( (a**2 + b**2 -c**2)/(2*a*b) )
		min_index = cos_theta.index( min(cos_theta) )
		sorted_points.append( points.pop( min_index ) )
	sorted_points.append( points.pop(0) )
	
	return sorted_points


def point_in_boundary(sorted_points, point):
    ''' sorted_points must be sorted by rotation '''
    spts = sorted_points

    ext_spts = spts + [spts[0]]
    angle_sum = 0
    for i, p1 in enumerate( ext_spts[:-1] ):
        p2 = ext_spts[i+1]
        #print i, p1, p2

        a = distance(point, p1)
        b = distance(point, p2)
        c = distance(p1, p2)

        if a == 0 or b == 0:
            angle_sum = 2*pi
            break
        else:
            #print sc.arccos( (a**2 + b**2 - c**2)/(2*a*b) )
            angle_sum += sc.arccos( (a**2 + b**2 - c**2)/(2*a*b) ) 

    #print angle_sum
    if abs(angle_sum - 2*pi) < pi/100000:
        return True
    else:
        return False


def test_point_in_boundary():
    sorted_points = [(0,0),(0.5,0),(0.5,1),(0,1)]
    point = (0,0)
    print point_in_boundary(sorted_points, point)


def area_points(sorted_points):
    ''' sorted_points must be sorted by rotation '''
    spts = sorted_points
    #print spts
    area = 0
    p1 = spts[0]
    for i, p2 in enumerate( spts[1:-1] ):
        p3 = spts[i+2]
        a = distance(p1, p2)
        b = distance(p1, p3)
        c = distance(p2, p3)
        s = 0.5*(a + b + c)
        area += sc.sqrt( s*(s-a)*(s-b)*(s-c) ) 
        #print p1, p2, p3
        #print 'a=%g, b=%g, c=%g' %(a, b, c)
        #print 's=%g, area=%g' %(s, area)

    return area


def test_area_points():
    #sorted_points = [(0,0),(1,0),(1,1),(0,1)]
    sorted_points = [(0,0),(0.5,0),(0.5,0.5),(0,0.5)]
    print area_points(sorted_points)


if __name__ == '__main__':
    '''
    # test the rotate function
    from scipy import pi
    point = (2, 1, 1)
    rotation_angles = (0, 0, pi)
    print rotate(point, rotation_angles) 
    print rotate(point, rotation_angles)[0] 
    '''
    #test_point_in_boundary()
    test_area_points()
