#!/usr/bin/env python

import boostmpi as mpi
import sys

x1, x2, x3 = None, None, None

if mpi.rank == 0:
	x1, x2, x3 = 1.2, 1.3, 1.4
	#ans = raw_input('Question? ')
	#print ans

#mpi.world.barrier()
x1, x2, x3 = mpi.broadcast( mpi.world, root=0, value=(x1, x2, x3) )

print mpi.rank, x1, x2, x3

mpi.world.barrier()
if mpi.rank == 0:
	print 'Question? (Y/n) '
	ans = sys.stdin.readline()[:-1]
	print ans
