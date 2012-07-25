'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 7. 17
 last update  :

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.output import Output

import pypar as mpi
server = 0


class OutputMpi( Output ):
	def __init__( s, dataform, global_pt1, global_pt2, spatial_step=(1,1,1) ):
		s.myrank = mpi.rank()
		s.dataform = dataform 
		s.global_pt1 = global_pt1
		s.global_pt2 = global_pt2
		s.spatial_step = spatial_step
		s.step = spatial_step[0]


	def set_space( s, Space, Nx_list ):
		s.Nx_sum_list = list( (sc.array(Nx_list)[:]).cumsum() )
		s.Nx_sum_list.insert(0,0)

		s.gi1 = s.global_pt1[0]
		s.gi2 = s.global_pt2[0]
		if s.gi1 == None: s.gi1 = 1
		if s.gi2 == None: s.gi2 = s.Nx_sum_list[-1]

		s.participant_list = s.calc_participant_list()

		if s.myrank is not server:
			if s.myrank in s.participant_list:
				s.participant = True
				pt1, pt2 = s.calc_points()
				Output.__init__( s, s.dataform, pt1, pt2, s.spatial_step )
			else:
				s.participant = False

		if s.myrank is not server:
			if s.myrank in s.participant_list:
				Output.set_space( s, Space )


	def calc_participant_list( s ):
		s.gi2 = s.gi2 - (s.gi2-s.gi1)%s.step

		for rank in xrange( 1, mpi.size() ):
			N1 = s.Nx_sum_list[rank-1]
			N2 = s.Nx_sum_list[rank] 
			if s.gi1 >= N1 and s.gi1 <= N2: 
				start_rank = rank
			if s.gi2 >= N1 and s.gi2 <= N2: 
				end_rank = rank

		return range( start_rank, end_rank+1 )


	def calc_points( s ):
		step_end = []
		for end in s.Nx_sum_list:
			step_end.append( end - (end-s.gi1)%s.step )

		step_start = list( sc.array( step_end )[:-1] + s.step )
		step_start.insert( 0, 0 )
		
		if s.myrank == s.participant_list[0]: i1 = s.gi1
		else: i1 = step_start[s.myrank] 
		i1 -= s.Nx_sum_list[s.myrank-1]
		if s.myrank == s.participant_list[-1]: i2 = s.gi2
		else: i2 = step_end[s.myrank]
		i2 -= s.Nx_sum_list[s.myrank-1]

		pt1 = list_replace( list(s.global_pt1), 0, i1 )
		pt2 = list_replace( list(s.global_pt2), 0, i2 )

		return pt1, pt2

	
	def send( s ):
		if s.participant:
			mpi.send( s.get_data(), 0 )


	def get_shape( s, gNx, gNy, gNz ):
		pt1, pt2 = list(s.global_pt1), list(s.global_pt2)
		for i in xrange(3):
			if pt1[i] == None: pt1[i] = 1
			if pt2[i] == None: pt2[i] = [gNx, gNy, gNz][i]

		shape = []
		for i in xrange(3):
			N = (pt2[i] - pt1[i])/s.spatial_step[i] + 1
			if N != 1: shape.append( N )

		return shape


	def gather( s ):
		data = mpi.receive( s.participant_list[0] )
		for rank in s.participant_list[1:]:
			data = sc.concatenate( (data, mpi.receive( rank )), axis=0 )

		return data
