#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : ecc_structure_2d.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 8. 27

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the structure functions using ec-conformal method.
	set_dielectric_line_array

===============================================================================
"""

from kufdtd.kufdtd_base import *

def ep_eff_sqrt_conformal(ep1, ep2, d1, d2):
	return ( ( d1*sc.sqrt(ep1) + d2*sc.sqrt(ep2) )**2 )/(d1 + d2)


def set_dielectric_line_arrays(structures, line_arrays, origin_cells, ds):
	Nx, Ny = line_arrays[x_axis].shape
	Ox, Oy = origin_cells[0]+0.5, origin_cells[1]+0.5

	bg_epr = structures[0]
	line_arrays[x_axis][:,:] = bg_epr
	line_arrays[y_axis][:,:] = bg_epr

	for i in xrange(Nx):
		x = (i-Ox)*ds
		j_list = []
		for structure in structures[1:]:
			result = structure.calc_intersection_points(x, grid_direction=y_axis)
			if result == None:
				break
			else:
				y1, y2, theta1, theta2 = result

			grid_y1, grid_y2 = y1/ds + Oy, y2/ds + Oy

			if grid_y1 != grid_y2:
				j1, j2 = int(grid_y1), int(grid_y2)
				#print 'grid_y1=%g, grid_y2=%g' % (grid_y1, grid_y2)
				#print 'j1=%d, j2=%d' % (j1, j2)
				d1 = grid_y1 - j1
				d2 = grid_y2 - j2

				line_arrays[y_axis][i,j1+1:j2] = structure.epr
				line_arrays[y_axis][i,j1] = ep_eff_sqrt_conformal(bg_epr, structure.epr, d1, 1-d1)
				line_arrays[y_axis][i,j2] = ep_eff_sqrt_conformal(structure.epr, bg_epr, d2, 1-d2)

				j_list.append(j1)
				j_list.append(j2)

		for j in j_list: 
			if j_list.count(j) >= 2:
				print 'Error: There are two intersection point on one cell grid!!'
				print 'point (i,j): (%d, %d)' % (i,j)
				print j_list
				sys.exit()


	for j in xrange(Ny):
		y = (j-Oy)*ds
		i_list = []
		for structure in structures[1:]:
			result = structure.calc_intersection_points(y, grid_direction=x_axis)
			if result == None:
				break
			else:
				x1, x2, theta1, theta2 = result

			grid_x1, grid_x2 = x1/ds + Ox, x2/ds + Ox
			
			if grid_x1 != grid_x2:
				i1, i2 = int(grid_x1), int(grid_x2)
				d1 = grid_x1 - i1
				d2 = grid_x2 - i2
				
				line_arrays[x_axis][i1+1:i2,j] = structure.epr
				line_arrays[x_axis][i1,j] = ep_eff_sqrt_conformal(bg_epr, structure.epr, d1, 1-d1)
				line_arrays[x_axis][i2,j] = ep_eff_sqrt_conformal(structure.epr, bg_epr, d2, 1-d2)

				i_list.append(i1)
				i_list.append(i2)

		for i in i_list: 
			if i_list.count(i) >= 2:
				print 'Error: There are two intersection point on one cell grid!!'
				print 'point (i,j): (%d, %d)' % (i,j)
				print i_list
				sys.exit()


if __name__ == '__main__':
	ep_list = [(1.,1.), (4., 4.), (1., 4.), (2., 3.)]
	d1_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

	print '-'*65
	print 'ep1\tep2\td1\td2\t\tep_eff\tep_eff_numerical'
	print '-'*65

	for ep1, ep2 in ep_list:
		for d1 in d1_list: 
			#print '%g\t%g\t%g\t%g' % (ep1, ep2, d1, 1-d1)
			ep_eff = ep_eff_sqrt_conformal(ep1, ep2, d1, 1-d1)
			ep_eff2 = ep_eff_sqrt_conformal_numerical(ep1, ep2, d1, 1-d1)
			print '%g\t%g\t%g\t%g\t\t%.4f\t%.4f' % (ep1, ep2, d1, 1-d1, ep_eff, ep_eff2)
		print '-'*65


	ep1, ep2, ep3 = 2., 3., 2.
	d1, d2 = 0.2, 0.3
	d3 = 1 - (d1 + d2)
	ep_eff = ep_eff_sqrt_conformal(ep1, ep2, d1, d2)
	ep_eff1 = ep_eff_sqrt_conformal(ep_eff, ep3, d1+d2, d3)

	ep_eff = ep_eff_sqrt_conformal(ep2, ep3, d2, d3)
	ep_eff2 = ep_eff_sqrt_conformal(ep1, ep_eff, d1, d2+d3)
	print '3 region overlap'
	print 'ep1\tep2\tep3\td1\td2\td3\t\tep_eff1\tep_eff2'
	print '-'*75
	print '%g\t%g\t%g\t%g\t%g\t%g\t\t%.4f\t%.4f' % (ep1, ep2, ep3, d1, d2, d3, ep_eff, ep_eff2)

