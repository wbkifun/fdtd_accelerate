#!/usr/bin/env python

import numpy as np
import pyopencl as cl
from datetime import datetime
import utils


def get_dt(func, args):
	tmax = 1000

	t0 = datetime.now()
	for tstep in xrange(1, tmax+1): func(*args)
	dt0 = datetime.now() - t0
	dt = (dt0.seconds + dt0.microseconds * 1e-6) / tmax

	return dt


nxs = range(96, 480+1, 32)	# nx**2, 72.0 KiB ~ 1.76 MiB
gpu_devices = utils.get_gpu_devices()
utils.print_gpu_info(gpu_devices)
context, queues = utils.create_context_queues(gpu_devices)
queue = queues[0]
mf = cl.mem_flags

print('nx\tdtoh\t\thtod\t\tdtoh(pinned)\thtod(pinned)')
for nx in nxs:
#for nx in [480]:
	print(nx),
	f = np.random.rand(2*nx*nx).astype(np.float32)
	dbuf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
	hbuf = np.zeros(f.shape, dtype=f.dtype)

	# dtoh (pageable)
	#cl.enqueue_read_buffer(queue, dbuf, hbuf)
	#print(np.linalg.norm(f - hbuf))
	#assert np.linalg.norm(f - hbuf) == 0
	dt = get_dt(cl.enqueue_read_buffer, (queue, dbuf, hbuf))
	print('\t%1.6f' % dt),

	# htod (pageable)
	dbuf_w = cl.Buffer(context, mf.WRITE_ONLY, f.nbytes)
	dt = get_dt(cl.enqueue_write_buffer, (queue, dbuf_w, hbuf))
	print('\t%1.6f' % dt)

	'''
	# dtoh (pinned)
	pin_hbuf = cl.Buffer(context, mf.READ_WRITE | mf.ALLOC_HOST_PTR, f.nbytes)
	map_hbuf, evt = cl.enqueue_map_buffer(queue, pin_hbuf, cl.map_flags.WRITE, 0, f.shape, f.dtype, 'C')
	cl.enqueue_read_buffer(queue, dbuf, map_hbuf)
	print(np.linalg.norm(f - map_hbuf))
	#assert np.linalg.norm(f - map_hbuf) == 0
	#dt = get_dt(cl.enqueue_read_buffer, (queue, dbuf, map_hbuf))
	print('\t%1.6f' % dt)
	'''
