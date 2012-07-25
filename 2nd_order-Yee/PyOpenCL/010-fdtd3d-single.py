#!/usr/bin/env python

import sys
import pyopencl as cl
import numpy as np
from my_cl_utils import print_device_info, get_optimal_global_work_size


# Platform, Device, Context and Queue
devices = []
platforms = cl.get_platforms()
for platform in platforms:
	devices.extend(platform.get_devices())
#print_device_info(platforms, devices)

device = devices[0]
context = cl.Context((device,))
queue = cl.CommandQueue(context, device)


# Parameter setup
nx, ny, nz = 240, 256, 256		# 540 MB
#nx, ny, nz = 512, 480, 480		# 3.96 GB
#nx, ny, nz = 256, 480, 960
tmax, tgap = 200, 10

print '(%d, %d, %d)' % (nx, ny, nz),
total_bytes = nx*ny*nz*4*9
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
eh_gpus = ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu\
		= [cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f) for i in range(6)]
ce_gpus = cex_gpu, cey_gpu, cez_gpu\
		= [cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf) for i in range(3)]


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
imsh = plt.imshow(np.ones((nx,ny),'f').T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.005)
plt.colorbar()


# Main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
	prg.update_h(queue, (Gs,), (Ls,), *eh_gpus)
	prg.update_e(queue, (Gs,), (Ls,), *(eh_gpus + ce_gpus))
	prg.update_src(queue, (Gs,), (Ls,), np.float32(tstep), ez_gpu)

	if tstep % tgap == 0:
		print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
		sys.stdout.flush()

		cl.enqueue_read_buffer(queue, ez_gpu, f)
		imsh.set_array( f[:,:,nz/2].T**2 )
		plt.draw()

print('')
