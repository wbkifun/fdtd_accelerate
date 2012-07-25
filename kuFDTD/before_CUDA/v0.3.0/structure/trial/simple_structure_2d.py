#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : ecc_structure_2d.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 7. 23

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the structure functions using simple average method.
	set_dielectric_line_array

===============================================================================
"""

from kufdtd.kufdtd_base import *

def set_dielectric_line_arrays(structures, line_arrays, coord_origin_cells, ds):
	Nx, Ny = line_arrays[x_axis].shape
	Ox, Oy = coord_origin_cells[0], coord_origin_cells[1]

	bg_epr = structures[0]
	line_arrays[x_axis][:,:] = bg_epr
	line_arrays[y_axis][:,:] = bg_epr

	for i in xrange(Nx):
		for j in xrange(Ny):
			x = (i-Ox)*ds
			y = (j-Oy)*ds

			for structure in structures[1:]:
				if structure.is_in( (x,y) ):
					ep1 = structure.epr
				else:
					ep1 = bg_epr

				if structure.is_in( (x-ds,y) ):
					ep2_x = structure.epr
				else:
					ep2_x = bg_epr

				line_arrays[y_axis][i,j] = 0.5*(ep1 + ep2_x)

				if structure.is_in( (x,y-ds) ):
					ep2_y = structure.epr
				else:
					ep2_y = bg_epr

				line_arrays[x_axis][i,j] = 0.5*(ep1 + ep2_y)
