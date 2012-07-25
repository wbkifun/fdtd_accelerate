'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.base import FdtdSpace


base_dir = '%s/dim3/cpu' % base_dir


class CpuSpace( FdtdSpace ):
	def __init__( s, Nx, Ny, Nz, dx, Ncore ):
		FdtdSpace.__init__( s, Nx, Ny, Nz, dx )

		s.Ncore = Ncore
		s.Ntot = Nx*Ny*Nz


	def verify_4xNz( s, Nz ):
		R = Nz%4
		if R != 0:
			print '-'*47
			print 'Error: Nz is not a multiple of 4.'
			print 'Recommend Nz: %d or %d' % (Nz-R, Nz-R+4)
			sys.exit(0)
