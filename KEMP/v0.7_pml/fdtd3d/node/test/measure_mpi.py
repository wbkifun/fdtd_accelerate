import numpy as np
import pyopencl as cl
from mpi4py import MPI

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.gpu import Fields, Core, Pbc, IncidentDirect, GetFields
from kemp.fdtd3d import node, cpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


is_plot = False

#nx, ny, nz = 240, 256, 256  # 540 MB
#nx, ny, nz = 544, 544, 544  # 5527 MB
#nx, ny, nz = 512, 512, 512  # 4608 MB
#nx, ny, nz = 480, 480, 480  # 3796 MB
nx, ny, nz = 1200, 256, 256  # 576 MB
#nx, ny, nz = 128, 128, 128  # 72 MB

coeff_use = 'e'
precision_float = 'single'

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]
fields = Fields(context, device, nx, ny, nz, coeff_use, precision_float)
#qtask = cpu.QueueTask()
#fields = Fields(context, device, qtask, nx, ny, nz, coeff_use, precision_float)
Core(fields)

tmax = 250 if is_plot else 1000
if rank == 0: direction = '+'
elif rank == size - 1: direction = '-'
else: direction = '+-'

exch = node.ExchangeMpiNonBlock(fields, direction)

#exch = node.ExchangeMpiBufferBlock(fields, direction)
#exch = node.ExchangeMpiBufferBlockSplit(fields, direction)
#exch = node.ExchangeMpiBufferNonBlockSplitEnqueue(fields, direction, tmax)
#if '+' in direction: cpu.Core(exch.cpuf_p)
#if '-' in direction: cpu.Core(exch.cpuf_m)

is_master = True if rank == 0 else False

if is_plot:
    Pbc(fields, 'yz')
    getf = GetFields(fields, 'ez', (0, 0, 0.5), (-1, -1, 0.5))

    tfunc = lambda tstep: 40 * np.sin(0.05 * tstep)
    if rank < size - 1:
        IncidentDirect(fields, 'ez', (220, 0.5, 0), (220, 0.5, -1), tfunc) 
    #if rank > 0:
    #    IncidentDirect(fields, 'ez', (20, 0.5, 0), (20, 0.5, -1), tfunc) 

    if is_master:
        # plot
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(figsize=(12,8))
        arr = np.zeros((size * nx, ny), fields.dtype)
        for i in range(1, size):
            plt.plot((i*nx, i*nx), (0, ny), color='k', linewidth=1.2)
        imag = plt.imshow(arr.T, interpolation='nearest', origin='lower', vmin=-0.7, vmax=0.7)
        plt.colorbar()

else: 
    getf = GetFields(fields, 'ez', (0, 0, 0), (0, 0, 0))


if is_master:
    print 'ns', fields.ns
    print 'nbytes (MB)', nx*ny*nz * 9 * 4. / (1024**2)

    from datetime import datetime
    from time import time, sleep
    t0 = datetime.now()
    t00 = time()

# main loop
for tstep in xrange(1, tmax+1):
    fields.update_e()
    exch.update_e()

    fields.update_h()
    exch.update_h()

    if tstep % 10 == 0 and is_plot:
        getf.get_event().wait()
        np.save('rank%d_%d' % (rank, tstep), getf.get_fields())

        if is_master:
            print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
            sys.stdout.flush()

            for i in range(size):
                load_fail = True
                while load_fail:
                    try:
                        arr[i*nx:(i+1)*nx,:] = np.load('rank%d_%d.npy' % (i, tstep))
                        load_fail = False
                    except:
                        sleep(0.1)

            imag.set_array(arr.T)
            #plt.savefig('./png/%.6d.png' % tstep)
            plt.draw()

if is_plot:
    if is_master: sleep(1)
else:
    getf.get_event().wait()
    #fields.enqueue_barrier()

if is_master:
    dt = time() - t00
    print dt
    print('[%s] %d/%d (%d %%) %f Mpoint/s' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100, nx*ny*nz*tmax/dt/1e6) )
