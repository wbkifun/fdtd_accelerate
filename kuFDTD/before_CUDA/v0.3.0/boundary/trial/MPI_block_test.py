#!/usr/bin/env python
#-*- coding: utf_8 -*-

import pypar as mpi

def mpicoord2rank(Nmpi, mpicoord):
    iDim = len(Nmpi)
    coord = []
    NumMpi = []
    for i in xrange(iDim):
	coord.append(mpicoord[i])
	NumMpi.append(Nmpi[i])
    if iDim ==2:
	coord.append(0)
	NumMpi.append(1)

    if (0<= coord[0] <NumMpi[0]) and (0<= coord[1] <NumMpi[1]) and\
	(0<= coord[2] <NumMpi[2]): 
	return coord[0] + Nmpi[0]*coord[1] + Nmpi[0]*Nmpi[1]*coord[2] +1
    else:
	return None

def rank2mpicoord(Nmpi, rank):

	iDim	= len(Nmpi)
	coord	= []
	NumMpi	= []
	for i in xrange(iDim):
		NumMpi.append(Nmpi[i])
	if iDim == 2:	NumMpi.append(1)
	k	= (int)(rank - 1)/(NumMpi[0]*NumMpi[1])
	if iDim == 2:	k = 0
	j	= (int)(rank - 1 - NumMpi[0]*NumMpi[1]*k)/NumMpi[0]
	i	= (int)(rank - 1 - NumMpi[0]*NumMpi[1]*k - NumMpi[0]*j)
	
	if iDim == 2:	return (i,j)
	elif iDim == 3:	return (i,j,k)


def calc_mpitarget(Nmpi, myrank):
    i,j,k = rank2mpicoord(Nmpi,myrank)

    part1 = mpicoord2rank(Nmpi,(i-1,j  ,k  ))
    part2 = mpicoord2rank(Nmpi,(i+1,j  ,k  ))
    part3 = mpicoord2rank(Nmpi,(i  ,j-1,k  ))
    part4 = mpicoord2rank(Nmpi,(i  ,j+1,k  ))
    part5 = mpicoord2rank(Nmpi,(i  ,j  ,k-1))
    part6 = mpicoord2rank(Nmpi,(i  ,j  ,k+1))

    target_list = (part1, part2, part3, part4, part5, part6)
    return target_list

def calc_mpitarget_pbc(Nmpi, myrank,pbc_opt=None):

    pbc_list = [None, None, None, None, None, None]

    part1, part2, part3, part4, part5, part6\
    = calc_mpitarget(Nmpi, myrank)

    i,j,k = rank2mpicoord(Nmpi,myrank)
    Nmpix, Nmpiy, Nmpiz = Nmpi

    PBC = pbc_opt
    if PBC != None:
	if 'x' in PBC:
	    if part1 == None:
		pbc_list[0] = mpicoord2rank(Nmpi,(Nmpix-1,j,k))

	    if part2 == None:
		pbc_list[1] = mpicoord2rank(Nmpi,(0,j,k))

	if 'y' in PBC:
	    if part3 == None:
		pbc_list[2] = mpicoord2rank(Nmpi,(i,Nmpiy-1,k))

	    if part4 == None:
		pbc_list[3] = mpicoord2rank(Nmpi,(i,0,k))

	if 'z' in PBC:
	    if part5 == None:
		pbc_list[4] = mpicoord2rank(Nmpi,(i,j,Nmpiz-1))

	    if part6 == None:
		pbc_list[5] = mpicoord2rank(Nmpi,(i,j,0))

    return pbc_list

def making_message_range(num_mpi_axis=3):
    message_range = []
    for D in xrange(num_mpi_axis*2):
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
	
    return message_range

def mpi_sendrecv(target_array, message_range, target_rank, tag_mark):

    send_range = message_range[0]
    recv_range = message_range[1]
    mpi.send(target_array[send_range].copy(), target_rank, tag=tag_mark)
    target_array[recv_range] = mpi.receive(target_rank,tag=tag_mark)

def mpi_recvsend(target_array, message_range, target_rank, tag_mark):

    send_range = message_range[0]
    recv_range = message_range[1]
    target_array[recv_range] = mpi.receive(target_rank,tag=tag_mark)
    mpi.send(target_array[send_range].copy(), target_rank, tag=tag_mark)

def mpi_exchange(arrays, Nmpi, myrank, target_list, all_message_range):
    num_array = len(arrays) # number of arrays
    mympicoord = rank2mpicoord(Nmpi, myrank)
    mpi_method = [mpi_sendrecv, mpi_recvsend] # MPI method function name

    for i in [0,2,4]: #three direction x, y, z
	jj = []
	jj.append(mympicoord[i/2] % 2) # odd or even -- first mark 
	jj.append((jj[0] + 1) % 2) # odd or even -- sencond mark

	for j in jj:  # odd : [1, 0]; even : [0, 1]
	    k = i + j # index of target_list
	    message_range = all_message_range[k] 
	    target_rank = target_list[k]

	    if target_rank != None:
		mpi_f = mpi_method[j] # SendRecv or RecvSend
		for l in xrange(num_array):
		    mpi_f(arrays[l], message_range[l], target_rank, l)

def mpi_exchange_pbc(arrays, target_list_pbc, all_message_range,\
	pbc_opt=None):

    if pbc_opt != None:
	num_array = len(arrays)
	for D, axis in enumerate('xyz'):
	    if axis in pbc_opt:
		left = target_list_pbc[2*D]
		right = target_list_pbc[2*D + 1]
		if left == None and right == None:
		    pass
		elif left == right:
		    message_range_le = all_message_range[2*D] #left
		    message_range_ri = all_message_range[2*D + 1] #right
		    for l in xrange(num_array):
			send_range_le = message_range_le[l][0]
			recv_range_le = message_range_le[l][1]
			send_range_ri = message_range_ri[l][0]
			recv_range_ri = message_range_ri[l][1]
			arrays[l][recv_range_ri] = arrays[l][send_range_le]
			arrays[l][recv_range_le] = arrays[l][send_range_ri]
		elif left != None:
		    message_range = all_message_range[2*D]
		    for l in xrange(num_array):
			mpi_sendrecv(arrays[l], message_range[l], left, l)
		elif right != None:
		    message_range = all_message_range[2*D + 1]
		    for l in xrange(num_array):
			mpi_recvsend(arrays[l], message_range[l], right, l)



# Bellow : Test Suite

def test_making_message_range(num_mpi_axis=3):
    message_range = []
    for D in xrange(num_mpi_axis*2):
	send_range = []
	recv_range = []
	message_range_temp = []
	send_range.append( (slice(None), slice(None), D) )
	send_range.append( (slice(None), slice(None), D) )
	send_range.append( (slice(None), slice(None), D) )
	recv_range.append( (slice(None), slice(None), D + 6) )
	recv_range.append( (slice(None), slice(None), D + 6) )
	recv_range.append( (slice(None), slice(None), D + 6) )
	for i in xrange(3):
	    message_range_temp.append([send_range[i], recv_range[i]])
	message_range.append(message_range_temp)

    return message_range

def test_mpi_sendrecv(target_array, message_range, target_rank, tag_mark):

    send_range = message_range[0]
    recv_range = message_range[1]
    mpi.send(target_array[send_range].copy(), target_rank, tag=tag_mark)
  #  print 'I`m', myrank, 'Send : ', target_rank, 'range : ', send_range
    target_array[recv_range] = mpi.receive(target_rank,tag=tag_mark)
  #  print 'I`m', myrank, 'Recv : ', target_rank, 'range : ', recv_range

def test_mpi_recvsend(target_array, message_range, target_rank, tag_mark):

    send_range = message_range[0]
    recv_range = message_range[1]
    target_array[recv_range] = mpi.receive(target_rank,tag=tag_mark)
  #  print 'I`m', myrank, 'Recv : ', target_rank, 'range : ', recv_range
    mpi.send(target_array[send_range].copy(), target_rank, tag=tag_mark)
  #  print 'I`m', myrank, 'Send : ', target_rank, 'range : ', send_range

def test_mpi_exchange(arrays, Nmpi, myrank, target_list, all_message_range):
    num_array = len(arrays) # number of arrays
    mympicoord = rank2mpicoord(Nmpi, myrank)
    mpi_method = [test_mpi_sendrecv, test_mpi_recvsend] # MPI method function name

    #print 'I`m ',myrank, 'Number of arrays : ', num_array
    #print 'I`m ',myrank, 'MPI Dimension : ', num_mpi_dim
    #print 'I`m ',myrank, 'MPI Coordinate : ', mympicoord
    #print

    for i in [0,2,4]: #three direction x, y, z
	jj = []
	jj.append(mympicoord[i/2] % 2) # odd or even -- first mark 
	jj.append((jj[0] + 1) % 2) # odd or even -- sencond mark

	#print 'I`m ', myrank, 'JJ : ',jj
	#print 'I`m ', myrank, 'I : ',i
	#print 'I`m ', myrank, 'targets : ',target_list
	#print
	for j in jj:  # odd : [1, 0]; even : [0, 1]
	    k = i + j # index of target_list
	    message_range = all_message_range[k] 
	    #print 'I`m ', myrank, 'J : ',j
	    #print 'I`m ', myrank, 'K : ',k
	    #print 'I`m ', myrank, 'message_range : ',message_range

	    target_rank = target_list[k]

	    if target_rank != None:
	        #print 'I`m ', myrank, 'target : ',target_rank
		mpi_f = mpi_method[j] # SendRecv or RecvSend
		for l in xrange(num_array):
		    mpi_f(arrays[l], message_range[l], target_rank, l)

	    #print 'I`m ', myrank, 'J : ',j, 'Done'

def test_mpi_exchange_pbc(arrays, myrank,target_list_pbc, all_message_range,\
	pbc_opt=None):

    if pbc_opt != None:
	#print pbc_opt
	num_array = len(arrays)
	for D, axis in enumerate('xyz'):
	    #print D, axis
	    if axis in pbc_opt:
		left = target_list_pbc[2*D]
		right = target_list_pbc[2*D + 1]
		#print 'I`m',myrank, axis, 'left:',left,'right:',right
		if left == None and right == None:
		    pass
		elif left == right:
		    #print 'same'
		    #print 'left',left,'right',right
		    message_range_le = all_message_range[2*D] #left
		    message_range_ri = all_message_range[2*D + 1] #right
		    for l in xrange(num_array):
			send_range_le = message_range_le[l][0]
			recv_range_le = message_range_le[l][1]
			send_range_ri = message_range_ri[l][0]
			recv_range_ri = message_range_ri[l][1]
			arrays[l][recv_range_ri] = arrays[l][send_range_le]
			arrays[l][recv_range_le] = arrays[l][send_range_ri]
		elif left != None:
		    #print 'I`m',myrank, axis, 'sending left:',left
		    message_range = all_message_range[2*D]
		    for l in xrange(num_array):
			mpi_sendrecv(arrays[l], message_range[l], left, l)
		elif right != None:
		    #print 'I`m',myrank, axis, 'sending right:',right
		    message_range = all_message_range[2*D + 1]
		    for l in xrange(num_array):
			mpi_recvsend(arrays[l], message_range[l], right, l)

if __name__ == '__main__':

    import scipy as sc


    Nnode = mpi.size()
    myrank = mpi.rank()
    mpi.finalize()
