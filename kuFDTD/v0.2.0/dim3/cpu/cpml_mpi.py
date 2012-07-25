'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 7. 18
 last update  :

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.cpu.cpml import CpmlNonKappa

import pypar as mpi
myrank = mpi.rank()
Nnode = mpi.size()


class CpmlNonKappaMpi( CpmlNonKappa ):
	def __init__( s, Nx_list, Npml, apply_direction ):
		s.Nx_list = list( Nx_list )
		CpmlNonKappa.__init__( s, Npml, apply_direction )

		s.Nx_sum_list, s.Nx_reverse_sum_list = s.calc_Nx_sum_list()
		s.Nx_list.insert( 0, 0 )

		s.f_end_rank, s.b_start_rank = s.calc_participant_rank()
		if 'f' not in apply_direction[0]: s.f_end_rank = 0
		if 'b' not in apply_direction[0]: s.b_start_rank = Nnode
		if s.f_end_rank == s.b_start_rank:
			if myrank == 0: 
				print 'Cpml Error: f_end_rank is equal to b_start_rank.'
			sys.exit(0)

		s.Npmlx = s.calc_Npmlx()


	def calc_Nx_sum_list( s ):
		Nx_sum_list = list( (sc.array(s.Nx_list)[:]).cumsum() )
		Nx_sum_list.insert( 0, 0 )

		tmp = []
		for i in xrange( len(s.Nx_list) ): tmp.append( s.Nx_list[i] )
		tmp.reverse()
		Nx_reverse_sum_list = list( (sc.array(tmp)[:]).cumsum() )
		Nx_reverse_sum_list.reverse()
		Nx_reverse_sum_list.insert( 0, 0 )
		Nx_reverse_sum_list.append( 0 )

		return Nx_sum_list, Nx_reverse_sum_list


	def calc_participant_rank( s ):
		Nsum = s.Nx_sum_list
		Nrsum = s.Nx_reverse_sum_list

		for rank in xrange( 1, Nnode ):
			if Nsum[rank] >= s.Npml and Nsum[rank-1] < s.Npml:
				f_end_rank = rank
			if Nrsum[rank] >= s.Npml and Nrsum[rank+1] < s.Npml:
				b_start_rank = rank

		return f_end_rank, b_start_rank


	def calc_Npmlx( s ):
		Nsum = s.Nx_sum_list
		Nrsum = s.Nx_reverse_sum_list

		if myrank < s.f_end_rank: 
			Npmlx = s.Nx_list[myrank]
		elif myrank == s.f_end_rank: 
			Npmlx = s.Npml - Nsum[myrank-1]
		elif myrank > s.b_start_rank: 
			Npmlx = s.Nx_list[myrank]
		elif myrank == s.b_start_rank: 
			Npmlx = s.Npml - Nrsum[myrank+1]
		else:
			Npmlx = 0

		return Npmlx


	def allocate_psi ( s ):
		s.apply_direction = list_replace( s.apply_direction, 0, '' )
		CpmlNonKappa.allocate_psi( s )

		if myrank <= s.f_end_rank:
			s.apply_direction = list_replace( s.apply_direction, 0, 'f' )
		elif myrank >= s.b_start_rank:
			s.apply_direction = list_replace( s.apply_direction, 0, 'b' )
		else:
			s.apply_direction = list_replace( s.apply_direction, 0, '' )

		s.allocate_psix( s.Npmlx )


	def allocate_coeff( s ):
		CpmlNonKappa.allocate_coeff( s )

		if s.Npmlx == s.Npml or s.Npmlx == 0:
			pass
		else:
			s.bEx = sc.zeros( 2*(s.Npmlx+1), 'f' )
			s.aEx = sc.zeros( 2*(s.Npmlx+1), 'f' )
			s.bHx = sc.zeros( 2*(s.Npmlx+1), 'f' )
			s.aHx = sc.zeros( 2*(s.Npmlx+1), 'f' )

			Nsum = s.Nx_sum_list
			Nrsum = s.Nx_reverse_sum_list

			if myrank <= s.f_end_rank:
				pt0 = Nsum[myrank-1] + 1
				slE = slH = slice( pt0, pt0 + s.Npmlx )

				s.bEx[1:s.Npmlx+1] = s.bE[slE]
				s.aEx[1:s.Npmlx+1] = s.aE[slE]
				s.bHx[1:s.Npmlx+1] = s.bH[slH]
				s.aHx[1:s.Npmlx+1] = s.aH[slH]

			elif myrank >= s.b_start_rank:
				pt0 = Nrsum[myrank+1]
				slE = slice( -pt0 -s.Npmlx -1, -pt0 -1 )
				slH = slice( -pt0 -s.Npmlx, -pt0 )
				if myrank == Nnode-1: slH = slice( -s.Npmlx, None )

				s.bEx[s.Npmlx+1:-1] = s.bE[slE]
				s.aEx[s.Npmlx+1:-1] = s.aE[slE]
				s.bHx[s.Npmlx+2:] = s.bH[slH]
				s.aHx[s.Npmlx+2:] = s.aH[slH]
