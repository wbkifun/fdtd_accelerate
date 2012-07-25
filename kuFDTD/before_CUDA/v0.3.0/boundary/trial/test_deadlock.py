#!/usr/bin/env python
#-*- coding: utf_8 -*-

import MPI_block_test as MPI
import pypar as mpi
import scipy as sc

Nnode = mpi.size()
myrank = mpi.rank()

def test_lock(Nmpi,fields,pbc_opt=None):
    if myrank == 0:
	print 'PBC : %s, start' % pbc_opt
    mpi.barrier()
    for i in xrange(len(fields)):
	fields[i][:,:,:6] = 1.
	fields[i][:,:,6:] = 0.
	#print 'I`m', myrank,'Field %s Direction x1 sum before = '%i,fields[i][:,:,6].sum()
	#print 'I`m', myrank,'Field %s Direction x2 sum before = '%i,fields[i][:,:,7].sum()
	#print 'I`m', myrank,'Field %s Direction y1 sum before = '%i,fields[i][:,:,8].sum()
	#print 'I`m', myrank,'Field %s Direction y2 sum before = '%i,fields[i][:,:,9].sum()
	#print 'I`m', myrank,'Field %s Direction z1 sum before = '%i,fields[i][:,:,10].sum()
	#print 'I`m', myrank,'Field %s Direction z2 sum before = '%i,fields[i][:,:,11].sum()
    mpi.barrier()

    if myrank != 0:
	targets = MPI.calc_mpitarget(Nmpi, myrank)
	targets_pbc = MPI.calc_mpitarget_pbc(Nmpi, myrank, pbc_opt)
	message_range = MPI.test_making_message_range()
	MPI.test_mpi_exchange(fields, Nmpi, myrank, targets, message_range)
	MPI.test_mpi_exchange_pbc(fields, myrank,targets_pbc, message_range, pbc_opt)

	for i in xrange(len(fields)):
	    print 'I`m', myrank,'Field %s Direction x1 sum after = '%i,fields[i][:,:,6].sum()
	    print 'I`m', myrank,'Field %s Direction x2 sum after = '%i,fields[i][:,:,7].sum()
	    print 'I`m', myrank,'Field %s Direction y1 sum after = '%i,fields[i][:,:,8].sum()
	    print 'I`m', myrank,'Field %s Direction y2 sum after = '%i,fields[i][:,:,9].sum()
	    print 'I`m', myrank,'Field %s Direction z1 sum after = '%i,fields[i][:,:,10].sum()
	    print 'I`m', myrank,'Field %s Direction z2 sum after = '%i,fields[i][:,:,11].sum()
    mpi.barrier()
    if myrank == 0:
	print 'PBC : %s, Done' % pbc_opt
	print
	print
	print


if __name__ == '__main__':

    Nx = 1000
    Ny = 100
    fields = []


    fields.append(sc.zeros((Nx,Ny,12),'f'))
    fields.append(sc.zeros((Nx,Ny,12),'f'))
    fields.append(sc.zeros((Nx,Ny,12),'f'))

    Nmpi = (8,1,1)

    # one field
    #test_lock(Nmpi,fields[:1])
    #test_lock(Nmpi,fields[:1],'x')
    #test_lock(Nmpi,fields[:1],'xy')
    #test_lock(Nmpi,fields[:1],'xyz')

    # two fields
    #test_lock(Nmpi,fields[:2])
    #test_lock(Nmpi,fields[:2],'x')
    #test_lock(Nmpi,fields[:2],'xy')
    #test_lock(Nmpi,fields[:2],'xyz')

    # three fields
    #test_lock(Nmpi,fields[:3])
    #test_lock(Nmpi,fields[:3],'x')
    #test_lock(Nmpi,fields[:3],'xy')
    test_lock(Nmpi,fields[:3],'xyz')

    Nmpi = (2,4,1)

    # one field
    #test_lock(Nmpi,fields[:1])
    #test_lock(Nmpi,fields[:1],'x')
    #test_lock(Nmpi,fields[:1],'xy')
    #test_lock(Nmpi,fields[:1],'xyz')

    # two fields
    #test_lock(Nmpi,fields[:2])
    #test_lock(Nmpi,fields[:2],'x')
    #test_lock(Nmpi,fields[:2],'xy')
    #test_lock(Nmpi,fields[:2],'xyz')

    # three fields
    #test_lock(Nmpi,fields[:3])
    #test_lock(Nmpi,fields[:3],'x')
    #test_lock(Nmpi,fields[:3],'xy')
    #test_lock(Nmpi,fields[:3],'xyz')

    Nmpi = (2,2,2)

    # one field
    #test_lock(Nmpi,fields[:1])
    #test_lock(Nmpi,fields[:1],'x')
    #test_lock(Nmpi,fields[:1],'xy')
    #test_lock(Nmpi,fields[:1],'xyz')

    # two fields
    #test_lock(Nmpi,fields[:2])
    #test_lock(Nmpi,fields[:2],'x')
    #test_lock(Nmpi,fields[:2],'xy')
    #test_lock(Nmpi,fields[:2],'xyz')

    # three fields
    #test_lock(Nmpi,fields[:3])
    #test_lock(Nmpi,fields[:3],'x')
    #test_lock(Nmpi,fields[:3],'xy')
    #test_lock(Nmpi,fields[:3],'xyz')

    mpi.finalize()
