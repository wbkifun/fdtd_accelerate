#!/usr/bin/env python
#-*- coding: utf_8 -*-

"""
 <File Description>

 File Name : mpi.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 7. 10

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

def mpicoord2rank(Nmpi, mpicoord):
	dimension = len(Nmpi)

	list_Nmpi = list(Nmpi)
	list_mpicoord = list(mpicoord)

	if dimension == 2:
		list_mpicoord.append(0)
		list_Nmpi.append(1)

	if (0 <= list_mpicoord[0] < list_Nmpi[0]) and (0<= list_mpicoord[1] <list_Nmpi[1]) and (0<= list_mpicoord[2] <list_Nmpi[2]): 
		return list_mpicoord[0] + Nmpi[0]*list_mpicoord[1] + Nmpi[0]*Nmpi[1]*list_mpicoord[2] + 1
	else:
		return None


def rank2mpicoord(Nmpi, rank):
	dimension = len(Nmpi)
	list_Nmpi = list(Nmpi)
	
	if dimension == 2:	
		list_Nmpi.append(1)

	if dimension == 3:
		k = (int)(rank - 1)/(list_Nmpi[0]*list_Nmpi[1])
	elif dimension == 2:	
		k = 0

	j = (int)(rank - 1 - list_Nmpi[0]*list_Nmpi[1]*k)/list_Nmpi[0]
	i = (int)(rank - 1 - list_Nmpi[0]*list_Nmpi[1]*k - list_Nmpi[0]*j)

	if dimension == 3:
		return (i,j,k) 
	elif dimension == 2:	
		return (i,j)
	

def calc_mpitarget(Nmpi, myrank):
	dimension = len(Nmpi)

	if dimension == 3:
		i,j,k = rank2mpicoord(Nmpi,myrank)

		target1 = mpicoord2rank(Nmpi,(i-1,j  ,k  ))
		target2 = mpicoord2rank(Nmpi,(i+1,j  ,k  ))
		target3 = mpicoord2rank(Nmpi,(i  ,j-1,k  ))
		target4 = mpicoord2rank(Nmpi,(i  ,j+1,k  ))
		target5 = mpicoord2rank(Nmpi,(i  ,j  ,k-1))
		target6 = mpicoord2rank(Nmpi,(i  ,j  ,k+1))

		target_list = (target1, target2, target3, target4, target5, target6)

	elif dimension == 2:
		i,j = rank2mpicoord(Nmpi,myrank)

		target1 = mpicoord2rank(Nmpi,(i-1,j  ))
		target2 = mpicoord2rank(Nmpi,(i+1,j  ))
		target3 = mpicoord2rank(Nmpi,(i  ,j-1))
		target4 = mpicoord2rank(Nmpi,(i  ,j+1))

		target_list = (target1, target2, target3, target4)

	return target_list


def calc_mpitarget_pbc(Nmpi, myrank, pbc_opt=None):
	dimension = len(Nmpi)

	if dimension == 3:
		target1, target2, target3, target4, target5, target6 = calc_mpitarget(Nmpi, myrank)
				
		i,j,k = rank2mpicoord(Nmpi,myrank)
		Nmpix, Nmpiy, Nmpiz = Nmpi
		
		if pbc_opt != None:
			if 'x' in pbc_opt:
				if target1 == None:
					pbc1 = mpicoord2rank(Nmpi,(Nmpix-1,j,k))
					
				if target2 == None:
					pbc2 = mpicoord2rank(Nmpi,(0,j,k))
					
			if 'y' in pbc_opt:
				if target3 == None:
					pbc3 = mpicoord2rank(Nmpi,(i,Nmpiy-1,k))
					
				if target4 == None:
					pbc4 = mpicoord2rank(Nmpi,(i,0,k))
					
			if 'z' in pbc_opt:
				if target5 == None:
					pbc5 = mpicoord2rank(Nmpi,(i,j,Nmpiz-1))
					
				if target6 == None:
					pbc6 = mpicoord2rank(Nmpi,(i,j,0))
					
			pbc_list = (pbc1, pbc2, pbc3, pbc4, pbc5, pbc6)

	elif dimension == 2:
		target1, target2, target3, target4 = calc_mpitarget(Nmpi, myrank)
				
		i,j = rank2mpicoord(Nmpi,myrank)
		Nmpix, Nmpiy = Nmpi
		
		if pbc_opt != None:
			if 'x' in pbc_opt:
				if target1 == None:
					pbc1 = mpicoord2rank(Nmpi,(Nmpix-1,j))
					
				if target2 == None:
					pbc2 = mpicoord2rank(Nmpi,(0,j))
					
			if 'y' in pbc_opt:
				if target3 == None:
					pbc3 = mpicoord2rank(Nmpi,(i,Nmpiy-1))
					
				if target4 == None:
					pbc4 = mpicoord2rank(Nmpi,(i,0))
					
			pbc_list = (pbc1, pbc2, pbc3, pbc4)

	return pbc_list 


def make_message_range( num_arrays=3 ):
	message_range = []
	
	if num_arrays == 3:
		for D in xrange(6):
			D = D + 1
			send_range = []
			recv_range = []
			message_range_temp = []
		
			if D == 1 or D == 2:		# x-direction
				D = -4*(D/2) + D
				send_range.append( (D, slice(1,  -1), slice(1,  -1)) )
				send_range.append( (D, slice(1,None), slice(1,  -1)) )
				send_range.append( (D, slice(1,  -1), slice(1,None)) )
				D = -(D/2)
				recv_range.append( (D, slice(1,  -1), slice(1,  -1)) )
				recv_range.append( (D, slice(1,None), slice(1,  -1)) )
				recv_range.append( (D, slice(1,  -1), slice(1,None)) )
				
			elif D == 3 or D == 4:		# y-direction
				D = -4*(D/4) + D/2
				send_range.append( (slice(1,None), D, slice(1,  -1)) )
				send_range.append( (slice(1,  -1), D, slice(1,  -1)) )
				send_range.append( (slice(1,  -1), D, slice(1,None)) )
				D = -(D/4)
				recv_range.append( (slice(1,None), D, slice(1,  -1)) )
				recv_range.append( (slice(1,  -1), D, slice(1,  -1)) )
				recv_range.append( (slice(1,  -1), D, slice(1,None)) )
				
			elif D == 5 or D == 6:		# z-direction
				D = -4*(D/6) + D/3
				send_range.append( (slice(1,None), slice(1,  -1), D) )
				send_range.append( (slice(1,  -1), slice(1,None), D) )
				send_range.append( (slice(1,  -1), slice(1,  -1), D) )
				D = -(D/6)
				recv_range.append( (slice(1,None), slice(1,  -1), D) )
				recv_range.append( (slice(1,  -1), slice(1,None), D) )
				recv_range.append( (slice(1,  -1), slice(1,  -1), D) )
				
			for i in xrange(3):
				message_range_temp.append([send_range[i], recv_range[i]])
				message_range.append(message_range_temp)

	elif num_arrays == 2:
		for D in xrange(4):
			D = D + 1
			send_range = []
			recv_range = []
			message_range_temp = []
		
			if D == 1 or D == 2:		# x-direction
				D = -4*(D/2) + D
				send_range.append( (D, slice(1,  -1)) )
				send_range.append( (D, slice(1,None)) )
				D = -(D/2)
				recv_range.append( (D, slice(1,  -1)) )
				recv_range.append( (D, slice(1,None)) )
				
			elif D == 3 or D == 4:		# y-direction
				D = -4*(D/4) + D/2
				send_range.append( (slice(1,None), D) )
				send_range.append( (slice(1,  -1), D) )
				D = -(D/4)
				recv_range.append( (slice(1,None), D) )
				recv_range.append( (slice(1,  -1), D) )
				
			for i in xrange(2):
				message_range_temp.append([send_range[i], recv_range[i]])
				message_range.append(message_range_temp)
			
	elif num_arrays == 1:
		for D in xrange(4):
			D = D + 1
			send_range = []
			recv_range = []
			message_range_temp = []
		
			if D == 1 or D == 2:		# x-direction
				D = -4*(D/2) + D
				send_range.append( (D, slice(1,  -1)) )
				D = -(D/2)
				recv_range.append( (D, slice(1,  -1)) )
				
			elif D == 3 or D == 4:		# y-directio
				D = -4*(D/4) + D/2
				send_range.append( (slice(1,  -1), D) )
				D = -(D/4)
				recv_range.append( (slice(1,  -1), D) )
				
			for i in xrange(1):
				message_range_temp.append([send_range[i], recv_range[i]])
				message_range.append(message_range_temp)

	return message_range	# == message_range_target


def mpi_sendrecv(target_array, message_range_field, target, tag_mark):
	send_range = message_range_field[0]
	recv_range = message_range_field[1]
	pypar.send(target_array[send_range].copy(), target, tag=tag_mark)
	target_array[recv_range] = pypar.receive(target, tag=tag_mark)
	
	
def mpi_recvsend(target_array, message_range_field, target, tag_mark):
	send_range = message_range_field[0]
	recv_range = message_range_field[1]
	target_array[recv_range] = pypar.receive(target, tag=tag_mark)
	pypar.send(target_array[send_range].copy(), target, tag=tag_mark)

	
def mpi_exchange(arrays, Nmpi, myrank, targets, message_range_targets, dimension=3):
	num_arrays = len(arrays) # number of arrays
	mympicoord = rank2mpicoord(Nmpi, myrank)
	mpi_funcs = [mpi_sendrecv, mpi_recvsend] # MPI method function name
	
	for i in xrange(0, dimension*2, 2): # direction [0,2,4] 3D, [0,2] 2D
		jj = []
		jj.append(mympicoord[i/2] % 2) # odd or even -- first mark 
		jj.append((jj[0] + 1) % 2) # odd or even -- sencond mark
		
		for j in jj:  # odd : [1, 0]; even : [0, 1]
			k = i + j # index of target_list
			message_range_target = message_range_targets[k]	# == message_range_fields
			target = targets[k]
			
			if target != None:
				mpi_func = mpi_funcs[j] # SendRecv or RecvSend
				
				for l in xrange(num_arrays):
					mpi_func(arrays[l], message_range_target[l], target, l)
					

def mpi_exchange_pbc(arrays, targets_pbc, message_range_targets, pbc_opt=None, dimension=3):
	if pbc_opt != None:
		num_arrays = len(arrays)
		
		if dimension == 3:
			direction_string = 'xyz'
		elif dimension == 2:
			direction_string = 'xy'

		for D, axis in enumerate( direction_string ):
			if axis in pbc_opt:
				target_left = targets_pbc[2*D]
				target_right = targets_pbc[2*D + 1]
				
				if target_left == None and target_right == None:
					pass

				elif target_left == target_right:
					message_range_target_left = message_range_targets[2*D] #target_left
					message_range_target_right = message_range_targets[2*D + 1] #target_right
					
					for l in xrange(num_arrays):
						send_range_left = message_range_target_left[l][0]
						recv_range_left = message_range_target_left[l][1]
						send_range_right = message_range_target_right[l][0]
						recv_range_right = message_range_target_right[l][1]
						arrays[l][recv_range_right] = arrays[l][send_range_left]
						arrays[l][recv_range_left] = arrays[l][send_range_right]

				elif target_left != None:
					message_range_target = message_range_targets[2*D]
					for l in xrange(num_arrays):
						mpi_sendrecv(arrays[l], message_range_target[l], target_left, l)

				elif target_right != None:
					message_range_target = message_range_targets[2*D + 1]
					for l in xrange(num_arrays):
						mpi_recvsend(arrays[l], message_range_target[l], target_right, l)

