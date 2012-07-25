'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 7. 10
 last update  : 2009. 7. 21

 Copyright : GNU GPL
'''

from kufdtd.common import *
import pypar as mpi

Nnode = mpi.size() - 1
myrank = mpi.rank()
server = 0


def mpi_sendrecv( target_array, target_rank, tag_mark ):
	mpi.send( target_array[1,1:,1:].copy(), target_rank, tag=tag_mark ) 
	target_array[0,1:,1:] = mpi.receive( target_rank, tag=tag_mark )


def mpi_recvsend( target_array, target_rank, tag_mark ):
	target_array[-1,1:,1:] = mpi.receive( target_rank,tag=tag_mark )
	mpi.send( target_array[-2,1:,1:].copy(), target_rank, tag=tag_mark )


def calc_mpitarget( myrank ):
	if myrank == 1:
		target_list = [None, 2]
	elif myrank == Nnode:
		target_list = [Nnode-1, None]
	else:
		target_list = [myrank-1, myrank+1]

	return target_list


def calc_oddeven( myrank ):
	target_list = calc_mpitarget( myrank )
	if ( myrank%2 == 0 ):	# even
		mpi_func_list = [mpi_sendrecv, mpi_recvsend]
	else:
		target_list.reverse()
		mpi_func_list = [mpi_recvsend, mpi_sendrecv]

	return target_list, mpi_func_list


def mpi_exchange( Ey, Ez, myrank, target_list, mpi_func_list ):
	for i in [0,1]:
		if target_list[i] != None:
			mpi_func_list[i]( Ey, target_list[i], 0 )
			mpi_func_list[i]( Ez, target_list[i], 1 )


def mpi_exchange_pbc( Ey, Ez, myrank, pbc_opt=None ):
	if pbc_opt != None and 'x' in pbc_opt:
		if myrank == 1:
			mpi_sendrecv( Ey, Nnode, 0 )
			mpi_sendrecv( Ez, Nnode, 1 )
		elif myrank == Nnode:
			mpi_recvsend( Ey, 1, 0 )
			mpi_recvsend( Ez, 1, 1 )


def calc_node_length_x_list( length_x ):
	node_length_x = length_x/Nnode
	node_length_x_list = []
	for i in xrange( Nnode-1 ):
		node_length_x_list.append( node_length_x )
	node_length_x_list.append( length_x - node_length_x*(Nnode-1) )

	return node_length_x_list


if __name__ == '__main__':
	import scipy as sc

	Ny = 100
	Nz = 100

	Ey = sc.zeros((4, Ny, Nz),'f')
	Ez = sc.zeros((4, Ny, Nz),'f')

	pbc_opt=None
	if myrank == server:
		print 'PBC : %s, start' % pbc_opt
	mpi.barrier()

	Ey[:,:,:] = 0.
	Ez[:,:,:] = 0.
	Ey[1:3,:,:] = 1.
	Ez[1:3,:,:] = 1.
	mpi.barrier()


	if myrank != server:

		target_list, mpi_func_list = calc_oddeven( myrank )
		mpi_exchange( Ey, Ez, myrank, target_list, mpi_func_list )
		mpi_exchange_pbc( Ey, Ez, myrank, pbc_opt )
		print 'I`m', myrank,'Ey Direction x1 sum after = ', Ey[ 0,:,:].sum()
		print 'I`m', myrank,'Ey Direction x2 sum after = ', Ey[-1,:,:].sum()
		print 'I`m', myrank,'Ez Direction x1 sum after = ', Ez[ 0,:,:].sum()
		print 'I`m', myrank,'Ez Direction x2 sum after = ', Ez[-1,:,:].sum()

	mpi.barrier()

	if myrank == server:
		print 'PBC : %s, Done' % pbc_opt
		print
		print
		print

	pbc_opt='x'
	if myrank == server:
		print 'PBC : %s, start' % pbc_opt
	mpi.barrier()

	Ey[:,:,:] = 0.
	Ez[:,:,:] = 0.
	Ey[1:3,:,:] = 1.
	Ez[1:3,:,:] = 1.
	mpi.barrier()


	if myrank != server:
		target_list, mpi_func_list = calc_oddeven( myrank )
		mpi_exchange( Ey, Ez, myrank, target_list, mpi_func_list )
		mpi_exchange_pbc( Ey, Ez, myrank, pbc_opt )
		print 'I`m', myrank,'Ey Direction x1 sum after = ', Ey[ 0,:,:].sum()
		print 'I`m', myrank,'Ey Direction x2 sum after = ', Ey[-1,:,:].sum()
		print 'I`m', myrank,'Ez Direction x1 sum after = ', Ez[ 0,:,:].sum()
		print 'I`m', myrank,'Ez Direction x2 sum after = ', Ez[-1,:,:].sum()
		

	mpi.barrier()

	if myrank == server:
		print 'PBC : %s, Done' % pbc_opt
		print
		print
		print


	mpi.finalize()
