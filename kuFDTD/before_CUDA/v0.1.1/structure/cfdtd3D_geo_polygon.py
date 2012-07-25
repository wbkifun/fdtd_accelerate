#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : polygon.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 02. 26. Thu

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the polygon class.

===============================================================================
"""

from scipy import array

class Polygon:
    def __init__(self, dx, unit, matter, points, accuracy):
        s.dx = dx
        s.unit = unit
        s.matter	=	matter
        s.points	=	points
        s.planes	=	[]
        s.accuracy = accuracy

	
	def dimensionless(s, dx):
		unitT	=	s.unit/dx
		
		points2	=	[]
		for pt in s.points:
			pt	=	tuple( array(pt)*unitT ) 
			points2.append(pt)

		s.points	=	points2	


	def set_bound(s):
		x,y,z	=	[], [], []
		for pt in s.points:
			x.append(pt[0])
			y.append(pt[1])
			z.append(pt[2])
		s.bound_x	=	( int(min(x)),int(max(x))+2 )
		s.bound_y	=	( int(min(y)),int(max(y))+2 )
		s.bound_z	=	( int(min(z)),int(max(z))+2 )

	
	def check_include_negative_val(s):
		for pt in s.points:
			if pt[0] < 0 or pt[1] < 0 or pt[2] < 0:
				print 'Error: A point include the negative value.'
				break


	def pt_on_line(s, line, pt):
		pt1	=	line[0]
		pt2	=	line[1]

		denominator_x	=	(pt[0]-pt2[0])
		denominator_y	=	(pt[1]-pt2[1])
		denominator_z	=	(pt[2]-pt2[2])
		
		result	=	False

		if denominator_x != 0:
			grady	=	(pt[1]-pt1[1])/(pt[1]-pt2[1]) 
			gradz	=	(pt[2]-pt1[2])/(pt[2]-pt2[2]) 

		if (abs(gradx-grady) < accuracy) and (abs(gradx-gradz) < accuracy):
			return True
		else:
			return False
		if denominator_x != 0:
			gradx	=	(pt[0]-pt1[0])/denominator_x 
		grady	=	(pt[1]-pt1[1])/(pt[1]-pt2[1]) 
		gradz	=	(pt[2]-pt1[2])/(pt[2]-pt2[2]) 

		if (abs(gradx-grady) < accuracy) and (abs(gradx-gradz) < accuracy):
			return True
		else:
			return False


	def eval_plane(s, plane, pt):
		from scipy import cross, dot
		pt1	=	plane[0]
		pt2	=	plane[1]
		pt3	=	plane[2]

		V12	=	tuple( array(pt2) - array(pt1) )
		V13	=	tuple( array(pt3) - array(pt1) )

		Vn	=	tuple( cross(V12,V13) )

		V01	=	tuple( array(pt1) - array(pt) )

		return dot(V01,Vn)


	def pt_on_plane(s, plane, pt):
		if abs( s.eval_plane(plane, pt) ) < accuracy:
			return True
		else:
			return False


	def is_exist_plane(s,i,j,k):
		pt1	=	s.points[i]
		pt2	=	s.points[j]
		pt3	=	s.points[k]

		result	=	False
		for plane in s.planes:
			if (pt1 in plane) and (pt2 in plane) and (pt3 in plane):
				result = True

		return result		


	def is_intersect_plane(s,i,j,k):
		Npt	=	len(s.points)
		plane	=	(s.points[i], s.points[j], s.points[k])

		result	=	False
		for i in xrange(Npt):
			for j in xrange(i+1,Npt):
				pt1	=	s.points[i]
				pt2	=	s.points[j]

				if s.pt_on_plane(plane,pt1) or s.pt_on_plane(plane,pt2):
					pass
				elif s.eval_plane(plane,pt1)*s.eval_plane(plane,pt2) < 0:
					result	=	True

		return result				


	def sort_plane_pts(s, plane):
		Npt		=	len(plane)
		plane2	=	[]

		if Npt == 3:
			plane2	=	plane

		elif Npt > 3:
			plane2.append( plane.pop(0) )
			while ( len(plane) != 0 ):
				tmp		=	[]
				for i in xrange(len(plane)):
					tmp.append( distance(plane2[-1], plane[i]) )
				
				short_index = tmp.index( min(tmp) )
				plane2.append( plane.pop(short_index) )

		else:
			print 'Invalid plane type'
		
		return plane2


	def coeff_plane(s,plane):
		d	=	s.eval_plane(plane, (0,0,0))
		c	=	s.eval_plane(plane, (0,0,1)) - d
		b	=	s.eval_plane(plane, (0,1,0)) - d
		a	=	s.eval_plane(plane, (1,0,0)) - d
		
		return (a,b,c,d)


	def make_planes(s):	
		print 'Make polygon planes from given points...' 
		print 'set dimensionless unit'
		s.dimensionless(s.dx)
		print 'set effective bound region'
		s.set_bound()
		s.check_include_negative_val()
		

		Npt	=	len(s.points)

		count	=	0
		for i in xrange(Npt):
			for j in xrange(i+1,Npt):
				for k in xrange(j+1,Npt):
					#count += 1
					#print 'count= %d, (%d,%d,%d)' %(count,i,j,k)

					pt1	=	s.points[i]
					pt2	=	s.points[j]
					pt3	=	s.points[k]
					#print pt1, pt2, pt3

					plane	=	[]
					#if s.pt_on_line((pt1,pt2), pt3):
					#	pass
					if s.is_exist_plane(i,j,k):	
						#print '\t[exist plane]'
						pass
					elif s.is_intersect_plane(i,j,k):	
						#print '\t[intersect plane]'
						pass
					else:
						#print '\t[correct plane]'
						plane.append(pt1)
						plane.append(pt2)
						plane.append(pt3)

						for z in xrange(k+1,Npt):
							pt4	=	s.points[z]
							if s.pt_on_plane((pt1,pt2,pt3), pt4):
								plane.append(pt4)

						plane = s.sort_plane_pts(plane)
						plane.append(s.coeff_plane(plane))
						s.planes.append(plane)
					#print '\n'
								
		Nplane	=	len(s.planes)
		print 'Number of planes: %d\n' %Nplane


	def print_planes(s):
		Nplane	=	len(s.planes)
		for i in xrange(Nplane):
			print s.planes[i]
			print ''


class box3D(polygon3D):
	def make_additional_points(s):
		x0, y0, z0	=	s.points[0][0], s.points[0][1], s.points[0][2]
		x1, y1, z1	=	s.points[1][0], s.points[1][1], s.points[1][2]
		dx, dy, dz	=	x1-x0, y1-y0, z1-z0

		points2	=	[]
		points2.append((x0   ,y0   ,z0   )) 
		points2.append((x0+dx,y0   ,z0   )) 
		points2.append((x0   ,y0+dy,z0   )) 
		points2.append((x0   ,y0   ,z0+dz)) 
		points2.append((x1   ,y1   ,z1   )) 
		points2.append((x1-dx,y1   ,z1   )) 
		points2.append((x1   ,y1-dy,z1   )) 
		points2.append((x1   ,y1   ,z1-dz)) 

		s.points	=	points2
