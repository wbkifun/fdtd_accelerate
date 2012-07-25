'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 7. 23
 last update  : 

 Copyright : GNU GPL
'''

from kufdtd.common import *


def is_overlap( c1, c2, d1, d2 ):
	d = abs(c1 - c2)

	if ( 2*d > d1 + d2 ): return False
	elif ( 2*d <= d1 + d2 ): return True



def is_overlap_3d( c1, c2, d1, d2 ):
	overlap_x = is_overlap( c1[0], c2[0], d1[0], d2[0] )
	overlap_y = is_overlap( c1[1], c2[1], d1[1], d2[1] )
	overlap_z = is_overlap( c1[2], c2[2], d1[2], d2[2] )

	return overlap_x*overlap_y*overlap_z



def find_overlap( N, overlap_group, structures ):
	new_overlap_group = []

	for s1 in overlap_group[N-1:]:
		for i, s2 in enumerate( structures ):
			c1 = s1.center_pt
			c2 = s2.center_pt
			d1 = s1.length
			d2 = s2.length
			if ( s1.matter_type is s2.matter_type ) and ( is_overlap_3d( c1, c2, d1, d2 ) ):
				new_overlap_group.append( structures.pop(i) )
	
	overlap_group += new_overlap_group

	return len(overlap_group), overlap_group
		


def calc_structure_groups( structures ):
	structure_groups = []
	while ( len(structures) > 0 ):
		Nin = 1
		Nout, glist = find_overlap( 0, [structures.pop(0)], structures )
		while ( Nin != Nout ):
			Nin = Nout
			Nout, glist = find_overlap( Nin, glist, structures )
		structure_groups.append( glist )

	wrapbox_groups = []
	for structure_group in structure_groups:
		st0 = structure_group[0]
		wrapbox_pt1 = list( st0.wrapbox_pt1 )
		wrapbox_pt2 = list( st0.wrapbox_pt2 )

		for st in structure_group[1:]:
			wrapbox_pt1[0] = min( wrapbox_pt1[0], st.wrapbox_pt1[0] )
			wrapbox_pt1[1] = min( wrapbox_pt1[1], st.wrapbox_pt1[1] )
			wrapbox_pt1[2] = min( wrapbox_pt1[2], st.wrapbox_pt1[2] )

			wrapbox_pt2[0] = max( wrapbox_pt2[0], st.wrapbox_pt2[0] )
			wrapbox_pt2[1] = max( wrapbox_pt2[1], st.wrapbox_pt2[1] )
			wrapbox_pt2[2] = max( wrapbox_pt2[2], st.wrapbox_pt2[2] )
		
		length = tuple( sc.array(wrapbox_pt2) - sc.array(wrapbox_pt1) )
		center_pt = tuple( sc.array(wrapbox_pt1) + sc.array(length)/2 )
		wrapbox_groups.append( [wrapbox_pt1, wrapbox_pt2, center_pt, length] )

	return structure_groups, wrapbox_groups 


def distance2( pt1, pt2 ):
	return ( ( sc.array(pt1) - sc.array(pt2) )**2 ).sum()


class Cylinder():
	def __init__( s, name, pt1_real, pt2_real, radius_real, matter_info ):
		s.name = name
		s.pt1_real = pt1_real
		s.pt2_real = pt2_real
		s.radius_real = radius_real
		s.matter_type = matter_info[0]
		s.matter_parameter = matter_info[1:]

		s.calc_wrapbox()
		
	
	def calc_wrapbox( s ):
		pt1 = s.pt1_real
		pt2 = s.pt2_real
		R = s.radius_real

		if pt1[0] != pt2[0]: 
			s.axial_direction = 'x'
			s.wrapbox_pt1 = ( pt1[0], pt1[1]-R, pt1[2]-R )
			s.wrapbox_pt2 = ( pt2[0], pt1[1]+R, pt1[2]+R )
			s.center_pt = ( pt1[0] + 0.5*(pt2[0]-pt1[0]), pt1[1], pt1[2] )

		elif pt1[1] != pt2[1]: 
			s.axial_direction = 'y'
			s.wrapbox_pt1 = ( pt1[0]-R, pt1[1], pt1[2]-R )
			s.wrapbox_pt2 = ( pt1[0]+R, pt2[1], pt1[2]+R )
			s.center_pt = ( pt1[0], pt1[1] + 0.5*(pt2[1]-pt1[1]), pt1[2] )

		elif pt1[2] != pt2[2]: 
			s.axial_direction = 'z'
			s.wrapbox_pt1 = ( pt1[0]-R, pt1[1]-R, pt1[2] )
			s.wrapbox_pt2 = ( pt1[0]+R, pt1[1]+R, pt2[2] )
			s.center_pt = ( pt1[0], pt1[1], pt1[2] + 0.5*(pt2[2]-pt1[2]) )

		s.length = tuple( sc.array(s.wrapbox_pt2) - sc.array(s.wrapbox_pt1) )
		s.center_pt = tuple( sc.array(s.wrapbox_pt1) + sc.array(s.length)/2 )
		
	
	def is_in( s, pt_real ):
		pt = pt_real
		pt1 = s.pt1_real
		pt2 = s.pt2_real
		R2 = s.radius_real**2
		
		if s.axial_direction == 'x':
			if pt[0] >= pt1[0] and pt[0] <= pt2[0] and distance2( pt[1:], pt1[1:] ) <= R2:
				return True
		elif s.axial_direction == 'y':
			if pt[1] >= pt1[1] and pt[1] <= pt2[1] and distance2( pt[::2], pt1[::2] ) <= R2:
				return True
		elif s.axial_direction == 'z':
			if pt[2] >= pt1[2] and pt[2] <= pt2[2] and distance2( pt[:-1], pt1[:-1] ) <= R2:
				return True

		return False
