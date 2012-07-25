#!/usr/bin/env python

import sys
import pyopencl as cl
import numpy as np
from my_cl_utils import print_device_info, get_optimal_global_work_size


# Platform, Device, Context and Queue
devices = []
queues = []
platforms = cl.get_platforms()
for platform in platforms:
	devices.extend(platform.get_devices())
#print_device_info(platforms, devices)

context = cl.Context(devices)
for device in devices:
	queues.append( cl.CommandQueue(context, device) )


# Parameter setup
ngpu = len(devices)
#nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
nx, ny, nz = 250, 480, 960
nnx = nx * ngpu
tmax, tgap = 200, 10

print '(%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx * ny * nz * 9 * np.nbytes['float32']
if total_bytes/(1024**3) == 0:
	print '%d MB' % ( total_bytes/(1024**2) )
else:
	print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

if nz%32 != 0:
	print "Error: nz is not multiple of 32"
	sys.exit()


# Allocation
f = np.zeros((nx,ny,nz), 'f')
cf = np.ones_like(f)*0.5

mf = cl.mem_flags
eh_gpus = []
ce_gpus = []
for i, queue in enumerate(queues):
	eh_gpus.append( [cl.Buffer(context, mf.READ_WRITE, f.nbytes) for m in range(6)] ) 
	ce_gpus.append( [cl.Buffer(context, mf.READ_ONLY, cf.nbytes) for m in range(3)] )
	for j in xrange(6):
		cl.enqueue_write_buffer(queue, eh_gpus[i][j], f)
	for j in xrange(3):
		cl.enqueue_write_buffer(queue, ce_gpus[i][j], cf)

b_offset = (nx-1) * ny * nz * np.nbytes['float32']
tmpfs = []
for i, queue in enumerate(queues):
	#tmpfs.append( [cl.Buffer(context, mf.READ_WRITE | mf.ALLOC_HOST_PTR, f.nbytes/nx) for m in range(2)] )
	tmpfs.append( [np.zeros((ny,nz), dtype=np.float32) for m in range(2)] )


# Program and Kernel
Ls = 256
Gs = get_optimal_global_work_size(device)
print('Ls = %d, Gs = %d' % (Ls, Gs))

kern = open('./fdtd3d.cl').read()
kernels = kern.replace('NXYZ',str(nx*ny*nz)).replace('NYZ',str(ny*nz)).replace('NX',str(nx)).replace('NY',str(ny)).replace('NZ',str(nz)).replace('DX',str(Ls))
#print kernels
prg = cl.Program(context, kernels).build()


# Plot
import matplotlib.pyplot as plt
plt.ion()
imsh = plt.imshow(np.ones((nnx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()
nf = np.zeros((nnx,ny), 'f')


# Main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	for queue, eh_gpu in zip(queues, eh_gpus):
		prg.update_h(queue, (Gs,), (Ls,), *eh_gpu)

	for queue, eh_gpu, tmpf in zip(queues, eh_gpus, tmpfs)[:-1]:
		cl.enqueue_read_buffer(queue, eh_gpu[4], tmpf[0], b_offset)	# hy_gpu
		cl.enqueue_read_buffer(queue, eh_gpu[5], tmpf[1], b_offset)	# hz_gpu
	for queue, eh_gpu, tmpf in zip(queues[1:], eh_gpus[1:], tmpfs[:-1]):
		cl.enqueue_write_buffer(queue, eh_gpu[4], tmpf[0])
		cl.enqueue_write_buffer(queue, eh_gpu[5], tmpf[1])

	for queue, eh_gpu, ce_gpu in zip(queues, eh_gpus, ce_gpus):
		prg.update_e(queue, (Gs,), (Ls,), *(eh_gpu + ce_gpu))

	for queue, eh_gpu, tmpf in zip(queues, eh_gpus, tmpfs)[1:]:
		cl.enqueue_read_buffer(queue, eh_gpu[1], tmpf[0])	# ey_gpu
		cl.enqueue_read_buffer(queue, eh_gpu[2], tmpf[1])	# ez_gpu
	for queue, eh_gpu, tmpf in zip(queues[:-1], eh_gpus[:-1], tmpfs[1:]):
		cl.enqueue_write_buffer(queue, eh_gpu[1], tmpf[0], b_offset)
		cl.enqueue_write_buffer(queue, eh_gpu[2], tmpf[1], b_offset)

	prg.update_src(queues[1], (Gs,), (Ls,), np.float32(tstep), eh_gpus[1][2])	# dev #1, ez_gpu

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		for i, queue in enumerate(queues):
			cl.enqueue_read_buffer(queue, eh_gpus[i][2], f)
			nf[i*nx:(i+1)*nx,:] = f[:,:,nz/2]
		imsh.set_array( nf.T**2 )
		plt.draw()

print('')
