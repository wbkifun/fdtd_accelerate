'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.base import FdtdSpace
import pycuda.driver as cuda


base_dir = '%s/dim3/gpu' % base_dir


class GpuSpace( FdtdSpace ):
	def __init__( s, Nx, Ny, Nz, dx ):
		FdtdSpace.__init__( s, Nx, Ny, Nz, dx )

		s.max_bpg = 65535

		s.kNx = sc.int32(s.Nx)
		s.kNy = sc.int32(s.Ny)
		s.kNz = sc.int32(s.Nz)


	def verify_16xNz( s ):
		R = s.Nz%16
		if ( R == 0 ):
			print '-'*47
			print 'Nz is a multiple of 16.'
		else:
			print '-'*47
			print 'Error: Nz is not a multiple of 16.'
			print 'Recommend Nz: %d or %d' % (s.Nz-R, s.Nz-R+16)
			sys.exit(0)


	def calc_occupancy( s, tpb ):
		# tpb: threads/block
		# tpw: threads/warp
		# wpb: warps/block
		# active_block:	active block/multiprocessor
		# active_warp:	active warp/multiprocessor
		
		tpw = 32
		max_active_block = 8	 
		max_active_warp = 32	 

		if ( tpb%tpw == 0 ): wpb = tpb/tpw
		else: wpb = tpb/tpw + 1

		if ( max_active_warp/wpb < max_active_block ): active_block = max_active_warp/wpb
		else: active_block = max_active_block

		active_warp = wpb*active_block
		occupancy = (float)(active_warp)/max_active_warp 

		return occupancy 


	def select_tpb( s, Ntot, Nsurplus_plane ):
		tpbs = [512, 256, 128] + range(511, 256, -1) + range(255, 128, -1) + range(127, 0, -1 )

		TPB, max_occupancy = 0, 0
		for tpb in tpbs:
			occupancy = s.calc_occupancy( tpb )

			if ( occupancy > max_occupancy ):
				max_occupancy = occupancy
				bpg = s.calc_bpg( Ntot, tpb )
				Nsurplus = tpb*bpg - Ntot

				if ( Nsurplus <= Nsurplus_plane ): 
					TPB = tpb

					if ( occupancy == 1 ):
						return TPB, occupancy

		if ( TPB == 0 ):
			print 'Error: There is not a TPB satisfied the conditions'
			sys.exit(0)

		return TPB, occupancy


	def calc_bpg( s, N, tpb ):
		if ( N%tpb == 0 ): bpg = N/tpb
		else: bpg = N/tpb + 1

		return bpg

	
	def calc_sub_bpgs( s, N, tpb ):
		bpg = s.calc_bpg( N, tpb )

		if ( bpg <= s.max_bpg ):
			return 1, [ bpg ], [ 0 ]

		else:
			sub_bpgs, idx0 = [], []

			nk = bpg/s.max_bpg + 1	# number of kernels
			sbpg = bpg/nk
			for i in xrange( nk ):
				sub_bpgs.append( sbpg )
				idx0.append( tpb*sbpg*i )
			sub_bpgs[-1] = sbpg + bpg%nk

			return nk, sub_bpgs, idx0
		


	def get_kernel_initmem( s ):
		fpath = '%s/core/initmem.cu' % base_dir
		mod = cuda.SourceModule( file( fpath,'r' ).read() )
		return mod.get_function("initmem")
