#!/usr/bin/env python
#-*- coding: utf_8 -*-

"""
 <File Description>

 File Name : mpi.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 7. 11

 Copyright : GNU LGPL

============================== < File Description > ===========================

Define the functions for MPI(Massage Passing Interface).
	int[or None] mpicoord2rank(number_nodes, mpicoord)
	tuple(int) rank2mpicoord(number_nodes, rank)
	tuple(int) calc_mpitarget(number_nodes, myrank)
	tuple(int) calc_mpitarget_pbc(number_nodes, myrank, pbc_opt=None)
	tuple(tuple(tuple(slice))) make_message_range(num_arrays=3)
	void mpi_sendrecv(target_array, message_range, target_rank, tag_mark)
	void mpi_recvsend(target_array, message_range, target_rank, tag_mark)
	void mpi_exchange(arrays, number_nodes, myrank, target_list, all_message_range)
	void mpi_exchange_pbc(arrays, target_list_pbc, all_message_range, pbc_opt=None)

===============================================================================
"""

import pypar

number_nodes = pypar.size()
myrank = pypar.rank()

class Mpi:
	def __init__(self, Nmpi, fields_list, pml_opt, pbc_opt):
		self.Nmpi = Nmpi
		self.efields = fields_list[0]
		self.hfields = fields_list[1]
		self.pml_opt = pml_opt
		self.pbc_opt = pbc_opt

		if self.efields == None:
			self.num_arrays = len(self.hfields)
		elif self.hfields == None:
			self.num_arrays = len(self.efields)

		self.dimension = len(self.Nmpi)
		self.mympicoord = self.rank2mpicoord(myrank)

		self.targets = self.calc_targets()
		self.targets_pbc = self.calc_targets_pbc()
		self.message_range_targets = self.calc_message_range_targets()
		self.mpi_funcs = [self.mpi_sendrecv, self.mpi_recvsend] # MPI send/receive functions

		# for calc_mpi_pml_opt, mpi_exchange_pbc methods
		if self.dimension == 3:
			self.direction_string = 'xyz'
		elif self.dimension == 2:
			self.direction_string = 'xy'

		self.mpi_pml_opt = self.calc_mpi_pml_opt()


	def mpicoord2rank(self, mpicoord):
		if self.dimension == 3:
			if (0 <= mpicoord[0] < self.Nmpi[0]) and (0 <= mpicoord[1] < self.Nmpi[1]) and (0 <= mpicoord[2] < self.Nmpi[2]): 
				return mpicoord[0] + self.Nmpi[0]*mpicoord[1] + self.Nmpi[0]*self.Nmpi[1]*mpicoord[2] + 1
			else:
				return None

		elif self.dimension == 2:
			if (0 <= mpicoord[0] < self.Nmpi[0]) and (0 <= mpicoord[1] < self.Nmpi[1]): 
				return mpicoord[0] + self.Nmpi[0]*mpicoord[1] + 1
			else:
				return None


	def rank2mpicoord(self, rank):
		if self.dimension == 3:
			k = (int)(rank - 1)/(self.Nmpi[0]*self.Nmpi[1])
			j = (int)(rank - 1 - self.Nmpi[0]*self.Nmpi[1]*k)/self.Nmpi[0]
			i = (int)(rank - 1 - self.Nmpi[0]*self.Nmpi[1]*k - self.Nmpi[0]*j)

			return (i,j,k) 

		elif self.dimension == 2:
			j = (int)(rank - 1)/self.Nmpi[0]
			i = (int)(rank - 1 - self.Nmpi[0]*j)

			return (i,j)
	

	def calc_targets(self):
		if self.dimension == 3:
			i,j,k = self.mympicoord

			target1 = self.mpicoord2rank((i-1,j  ,k  ))
			target2 = self.mpicoord2rank((i+1,j  ,k  ))
			target3 = self.mpicoord2rank((i  ,j-1,k  ))
			target4 = self.mpicoord2rank((i  ,j+1,k  ))
			target5 = self.mpicoord2rank((i  ,j  ,k-1))
			target6 = self.mpicoord2rank((i  ,j  ,k+1))

			targets = (target1, target2, target3, target4, target5, target6)

		elif self.dimension == 2:
			i,j = self.mympicoord

			target1 = self.mpicoord2rank((i-1,j  ))
			target2 = self.mpicoord2rank((i+1,j  ))
			target3 = self.mpicoord2rank((i  ,j-1))
			target4 = self.mpicoord2rank((i  ,j+1))

			targets = (target1, target2, target3, target4)

		return targets


	def calc_targets_pbc(self):
		if self.dimension == 3:
			target1, target2, target3, target4, target5, target6 = self.targets
			targets_pbc = [None, None, None, None, None, None]
					
			i,j,k = self.mympicoord
			Nmpix, Nmpiy, Nmpiz = self.Nmpi
			
			if self.pbc_opt != None:
				if 'x' in self.pbc_opt:
					if target1 == None:
						targets_pbc[0] = self.mpicoord2rank((Nmpix-1,j,k))
						
					if target2 == None:
						targets_pbc[1] = self.mpicoord2rank((0,j,k))
						
				if 'y' in self.pbc_opt:
					if target3 == None:
						targets_pbc[2] = self.mpicoord2rank((i,Nmpiy-1,k))
						
					if target4 == None:
						targets_pbc[3] = self.mpicoord2rank((i,0,k))
						
				if 'z' in self.pbc_opt:
					if target5 == None:
						targets_pbc[4] = self.mpicoord2rank((i,j,Nmpiz-1))
						
					if target6 == None:
						targets_pbc[5] = self.mpicoord2rank((i,j,0))
						
		elif self.dimension == 2:
			target1, target2, target3, target4 = self.targets
			targets_pbc = [None, None, None, None]
					
			i,j = self.mympicoord
			Nmpix, Nmpiy = self.Nmpi
			
			if self.pbc_opt != None:
				if 'x' in self.pbc_opt:
					if target1 == None:
						targets_pbc[0] = self.mpicoord2rank((Nmpix-1,j))
						
					if target2 == None:
						targets_pbc[1] = self.mpicoord2rank((0,j))
						
				if 'y' in self.pbc_opt:
					if target3 == None:
						targets_pbc[2] = self.mpicoord2rank((i,Nmpiy-1))
						
					if target4 == None:
						targets_pbc[3] = self.mpicoord2rank((i,0))

		return tuple(targets_pbc)


	def calc_mpi_pml_opt(self):
		if self.dimension == 3:
			mpi_pml_opt = ['', '', '']
		elif self.dimension == 2:
			mpi_pml_opt = ['', '']

		for D, axis in enumerate( self.direction_string ):
			if axis in self.pml_opt:
				if self.mympicoord[D] == 0 and self.targets_pbc[2*D] == None:
					mpi_pml_opt[D] += 'f'
				if self.mympicoord[D] == self.Nmpi[D]-1 and self.targets_pbc[2*D+1] == None:
					mpi_pml_opt[D] += 'b'

		return tuple(mpi_pml_opt)


	def calc_message_range_targets(self):
		message_range = []
		
		if self.num_arrays == 3:
			for D in xrange(6):
				D = D + 1
				send_range = []
				recv_range = []
				message_range_temp = []
			
				if D == 1 or D == 2:		# x-direction
					d = -4*(D/2) + D
					send_range.append( (d, slice(1,  -1), slice(1,  -1)) )
					send_range.append( (d, slice(1,None), slice(1,  -1)) )
					send_range.append( (d, slice(1,  -1), slice(1,None)) )
					d = -(D/2)
					recv_range.append( (d, slice(1,  -1), slice(1,  -1)) )
					recv_range.append( (d, slice(1,None), slice(1,  -1)) )
					recv_range.append( (d, slice(1,  -1), slice(1,None)) )
					
				elif D == 3 or D == 4:		# y-direction
					d = -4*(D/4) + D/2
					send_range.append( (slice(1,None), d, slice(1,  -1)) )
					send_range.append( (slice(1,  -1), d, slice(1,  -1)) )
					send_range.append( (slice(1,  -1), d, slice(1,None)) )
					d = -(D/4)
					recv_range.append( (slice(1,None), d, slice(1,  -1)) )
					recv_range.append( (slice(1,  -1), d, slice(1,  -1)) )
					recv_range.append( (slice(1,  -1), d, slice(1,None)) )
					
				elif D == 5 or D == 6:		# z-direction
					d = -4*(D/6) + D/3
					send_range.append( (slice(1,None), slice(1,  -1), d) )
					send_range.append( (slice(1,  -1), slice(1,None), d) )
					send_range.append( (slice(1,  -1), slice(1,  -1), d) )
					d = -(D/6)
					recv_range.append( (slice(1,None), slice(1,  -1), d) )
					recv_range.append( (slice(1,  -1), slice(1,None), d) )
					recv_range.append( (slice(1,  -1), slice(1,  -1), d) )
					
				for i in xrange(3):
					message_range_temp.append([send_range[i], recv_range[i]])

				message_range.append(message_range_temp)

		elif self.num_arrays == 2:
			for D in xrange(4):
				D = D + 1
				send_range = []
				recv_range = []
				message_range_temp = []
			
				if D == 1 or D == 2:		# x-direction
					d = -4*(D/2) + D
					send_range.append( (d, slice(1,  -1)) )
					send_range.append( (d, slice(1,None)) )
					d = -(D/2)
					recv_range.append( (d, slice(1,  -1)) )
					recv_range.append( (d, slice(1,None)) )
					
				elif D == 3 or D == 4:		# y-direction
					d = -4*(D/4) + D/2
					send_range.append( (slice(1,None), d) )
					send_range.append( (slice(1,  -1), d) )
					d = -(D/4)
					recv_range.append( (slice(1,None), d) )
					recv_range.append( (slice(1,  -1), d) )
					
				for i in xrange(2):
					message_range_temp.append([send_range[i], recv_range[i]])

				message_range.append(message_range_temp)
				
		elif self.num_arrays == 1:
			for D in xrange(4):
				D = D + 1
				send_range = []
				recv_range = []
				message_range_temp = []
			
				if D == 1 or D == 2:		# x-direction
					d = -4*(D/2) + D
					send_range.append( (d, slice(1,  -1)) )
					d = -(D/2)
					recv_range.append( (d, slice(1,  -1)) )
					
				elif D == 3 or D == 4:		# y-directio
					d = -4*(D/4) + D/2
					send_range.append( (slice(1,  -1), d) )
					d = -(D/4)
					recv_range.append( (slice(1,  -1), d) )
					
				for i in xrange(1):
					message_range_temp.append([send_range[i], recv_range[i]])

				message_range.append(message_range_temp)

		return message_range	# == message_range_target


	def mpi_sendrecv(self, array, message_range_field, target, tag_mark):
		send_range = message_range_field[0]
		recv_range = message_range_field[1]
		pypar.send(array[send_range].copy(), target, tag=tag_mark)
		array[recv_range] = pypar.receive(target, tag=tag_mark)
	
	
	def mpi_recvsend(self, array, message_range_field, target, tag_mark):
		send_range = message_range_field[0]
		recv_range = message_range_field[1]
		array[recv_range] = pypar.receive(target, tag=tag_mark)
		pypar.send(array[send_range].copy(), target, tag=tag_mark)

	
	def mpi_exchange(self, arrays):
		for i in xrange(0, self.dimension*2, 2): # direction [0,2,4] 3D, [0,2] 2D
			jj = []
			jj.append(self.mympicoord[i/2] % 2) # odd or even -- first mark 
			jj.append((jj[0] + 1) % 2) # odd or even -- sencond mark
			
			#print 'direction=%d, myrank=%d, jj=%s' % (i/2, myrank, jj)
			for j in jj:  # odd : [1, 0]; even : [0, 1]
				k = i + j # index of target_list
				target = self.targets[k]
				#print target
				
				if target != None:
					message_range_target = self.message_range_targets[k]	# == message_range_fields
					#print 'k=%d, myrank=%d' % (k, myrank), message_range_target
					mpi_func = self.mpi_funcs[j] # SendRecv or RecvSend
					
					for l in xrange(self.num_arrays):
						mpi_func(arrays[l], message_range_target[l], target, l)

		if self.pbc_opt != None:
			self.mpi_exchange_pbc(arrays)
					

	def mpi_exchange_efields(self):
		if self.efields == None:
			pass
		else:
			self.mpi_exchange(self.efields)


	def mpi_exchange_hfields(self):
		if self.hfields == None:
			pass
		else:
			self.mpi_exchange(self.hfields)


	def mpi_exchange_pbc(self, arrays):
		for D, axis in enumerate( self.direction_string ):
			if axis in self.pbc_opt:
				target_left = self.targets_pbc[2*D]
				target_right = self.targets_pbc[2*D + 1]
				
				if target_left == None and target_right == None:
					pass

				elif target_left == target_right:
					message_range_target_left = self.message_range_targets[2*D] #target_left
					message_range_target_right = self.message_range_targets[2*D + 1] #target_right
					
					for l in xrange(self.num_arrays):
						send_range_left = message_range_target_left[l][0]
						recv_range_left = message_range_target_left[l][1]
						send_range_right = message_range_target_right[l][0]
						recv_range_right = message_range_target_right[l][1]
						arrays[l][recv_range_right] = arrays[l][send_range_left]
						arrays[l][recv_range_left] = arrays[l][send_range_right]

				elif target_left != None:
					message_range_target = self.message_range_targets[2*D]
					for l in xrange(self.num_arrays):
						mpi_sendrecv(arrays[l], message_range_target[l], target_left, l)

				elif target_right != None:
					message_range_target = self.message_range_targets[2*D + 1]
					for l in xrange(self.num_arrays):
						mpi_recvsend(arrays[l], message_range_target[l], target_right, l)
