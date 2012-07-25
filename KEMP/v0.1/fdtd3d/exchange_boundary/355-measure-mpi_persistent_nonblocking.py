#!/usr/bin/env python

import numpy as np
import sys
from datetime import datetime

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


tmax = 100
start = 2 * np.nbytes['float32'] * (3 * 32)**2	# 96, 73728 B, 72.0 KIB
end = 2 * np.nbytes['float32'] * (15 * 32)**2	# 480, 1843200 B, 1.76 MiB
increment = (end - start) / 16
nbytes = np.arange(start, end+1, increment)
dts = np.zeros(nbytes.size)


# verify h5 file exist
if rank == 0:
	import os
	h5_path = './bandwidth_GbE_nonblocking.h5'
	if os.path.exists(h5_path):
		print('Error: File exist %s' % h5_path)
		sys.exit()


for i, nbyte in enumerate(nbytes):
	if rank == 0:
		dts[i] = comm.recv(source=1, tag=10)	# source, tag
		print('nbyte = %d, dt = %f' % (nbyte, dts[i]))

	elif rank == 1:
		arr_send = np.random.rand(nbyte/np.nbytes['float32']).astype(np.float32)
		arr_recv = np.zeros_like(arr_send)
		req_send = comm.Send_init(arr_send, dest=2, tag=10)
		req_recv = comm.Recv_init(arr_recv, source=2, tag=20)
		reqs = [req_send, req_recv]
		t0 = datetime.now()
		for tstep in xrange(1, tmax+1):
			for req in reqs: req.Start()
			for req in reqs: req.Wait()
		dt0 = datetime.now() - t0
		dt = (dt0.seconds + dt0.microseconds * 1e-6) / tmax
		#print('[%d] dt = %f' % (rank, dt))
		comm.send(dt, dest=0, tag=10)	# data, dest, tag

	elif rank == 2:
		arr_send = np.random.rand(nbyte/np.nbytes['float32']).astype(np.float32)
		arr_recv = np.zeros_like(arr_send)
		req_send = comm.Send_init(arr_send, dest=1, tag=20)
		req_recv = comm.Recv_init(arr_recv, source=1, tag=10)
		reqs = [req_send, req_recv]
		for tstep in xrange(1, tmax+1):
			for req in reqs: req.Start()
			for req in reqs: req.Wait()
		

# Save as h5
if rank == 0:
	import h5py as h5
	f = h5.File(h5_path, 'w')
	f.create_dataset('nbytes', data=nbytes)
	f.create_dataset('dts', data=dts)
