'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 2009. 7. 23

 Copyright : GNU GPL
'''

from kufdtd.common import *


class FdtdSpace:
	def __init__( s, Nx, Ny, Nz, dx ):
		s.Nx = Nx
		s.Ny = Ny
		s.Nz = Nz
		s.dx = dx
		s.N = ( Nx, Ny, Nz )

		courant = 0.5		# Courant factor
		s.dt = courant*dx/light_velocity

		s.bytes_f = sc.zeros(1,'f').nbytes
		s.mem_usage = 0


	def set_wrapbox( s, wrapbox_pt1, wrapbox_pt2, center_pt, length ):
		s.wrapbox_pt1 = wrapbox_pt1
		s.wrapbox_pt2 = wrapbox_pt2
		s.center_pt = center_pt
		s.length = length
