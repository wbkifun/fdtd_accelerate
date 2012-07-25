#!/usr/bin/env python

import numpy as np
import subprocess as sp

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stdout, stderr = sp.Popen(['hostname'], stdout=sp.PIPE).communicate()
print('rank = %d, %s' % (rank, stdout))

if rank == 0:
	a = 12.3
	lst = ['a', 4.56]
	arr = np.random.rand(3,5).astype(np.float32)

	print('rank %d: a = %g, lst = %s' % (rank, a, lst))
	print('arr = %s' % (arr))
	comm.send(a, 1, 10)
	comm.send(lst, 1, 11)
	comm.Send(arr, 1, 12)

elif rank == 1:
	a = comm.recv(source=0, tag=10)
	lst = comm.recv(source=0, tag=11)
	arr = np.zeros((3,5), dtype=np.float32)
	comm.Recv(arr, 0, 12)
	print('rank %d: a = %g, lst = %s' % (rank, a, lst))
	print('arr = %s' % (arr))
