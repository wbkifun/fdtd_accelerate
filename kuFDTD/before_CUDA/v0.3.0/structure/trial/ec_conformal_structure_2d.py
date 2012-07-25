#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : ecc_structure_2d.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 7. 2

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the structure functions using ec-conformal method.
	set_dielectric_line_array

===============================================================================
"""

from kufdtd.kufdtd_base import *

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
			j1, j2 = int(grid_y1), int(grid_y2)
			#print 'grid_y1=%g, grid_y2=%g' % (grid_y1, grid_y2)
			#print 'j1=%d, j2=%d' % (j1, j2)
			d1 = grid_y1 - j1
			d2 = grid_y2 - j2

			line_arrays[y_axis][i,j1+1:j2] = structure.epr
		
			d = d1
			ep1 = bg_epr
			ep2 = structure.epr
			epf = ep1/ep2
			si2 = sc.sin(theta1)**2
			co2 = sc.cos(theta1)**2
			line_arrays[y_axis][i,j1] = ( d*ep1+(1-d)*ep2*(epf*epf*si2+co2) )/( (d+(1-d)*epf)**2*si2+co2 )

			d = d2
			ep1 = structure.epr
			ep2 = bg_epr
			epf = ep1/ep2
			si2 = sc.sin(theta2)**2
			co2 = sc.cos(theta2)**2
			line_arrays[y_axis][i,j2] = ( d*ep1+(1-d)*ep2*(epf*epf*si2+co2) )/( (d+(1-d)*epf)**2*si2+co2 )

			j_list.append(j1)
			j_list.append(j2)

		for j in j_list: 
			if j_list.count(j) >= 2:
				print 'Error: There are two intersection point on one cell grid!!'
				print 'point (i,j): (%d, %d)' % (i,j)
				#sys.exit()


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
			i1, i2 = int(grid_x1), int(grid_x2)
			d1 = grid_x1 - i1
			d2 = grid_x2 - i2
			
			line_arrays[x_axis][i1+1:i2,j] = structure.epr

			d = d1
			ep1 = bg_epr
			ep2 = structure.epr
			epf = ep1/ep2
			si2 = sc.sin(theta1)**2
			co2 = sc.cos(theta1)**2
			line_arrays[x_axis][i1,j] = ( d*ep1+(1-d)*ep2*(epf*epf*si2+co2) )/( (d+(1-d)*epf)**2*si2+co2 )

			d = d2
			ep1 = structure.epr
			ep2 = bg_epr
			epf = ep1/ep2
			si2 = sc.sin(theta2)**2
			co2 = sc.cos(theta2)**2
			line_arrays[x_axis][i2,j] = ( d*ep1+(1-d)*ep2*(epf*epf*si2+co2) )/( (d+(1-d)*epf)**2*si2+co2 )

			i_list.append(i1)
			i_list.append(i2)

		for i in i_list: 
			if i_list.count(i) >= 2:
				print 'Error: There are two intersection point on one cell grid!!'
				print 'point (i,j): (%d, %d)' % (i,j)
				#sys.exit()
