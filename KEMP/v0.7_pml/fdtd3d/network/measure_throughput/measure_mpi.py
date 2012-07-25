import numpy as np
import pyopencl as cl
from mpi4py import MPI

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d import gpu, cpu, node, network


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


try:
    nx = int(sys.argv[1])
except:
    nx = 360

is_plot = False
ny, nz = 250, 256
coeff_use = 'e'
precision_float = 'single'
use_cpu_core = 1

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]
gpuf = gpu.Fields(context, device, nx, ny, nz, coeff_use, precision_float)

tmax = 250 if is_plot else 1000
#if rank == 0: direction = '+'
#elif rank == size - 1: direction = '-'
#else: direction = '+-'
direction = '+-'

buffer_dict = {}
if '+' in direction:
    buffer_dict['x+'] = cpu.Fields(cpu.QueueTask(), 2, ny, nz, coeff_use, precision_float, use_cpu_core)
if '-' in direction:
    buffer_dict['x-'] = cpu.Fields(cpu.QueueTask(), 2, ny, nz, coeff_use, precision_float, use_cpu_core)

nodef = node.Fields([gpuf], buffer_dict)
node.Core(nodef)

#network.ExchangeMpiBlock(nodef, direction)
#network.ExchangeMpiNonBlock(nodef, direction)
network.ExchangeMpiBuffer(nodef)


if is_plot:
    is_master = True if rank == 1 else False
    getf = gpu.GetFields(gpuf, 'ez', (0, 0, 0.5), (-1, -1, 0.5))

    tfunc = lambda tstep: 40 * np.sin(0.05 * tstep)
    if rank < size - 1:
        gpu.IncidentDirect(gpuf, 'ez', (220, 0.5, 0), (220, 0.5, -1), tfunc) 
    #if rank > 0:
    #    gpu.IncidentDirect(gpuf, 'ez', (20, 0.5, 0), (20, 0.5, -1), tfunc) 

    if is_master:
        # plot
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(figsize=(12,8))
        arr = np.zeros((size * nx, ny), gpuf.dtype)
        for i in range(1, size):
            plt.plot((i*nx, i*nx), (0, ny), color='k', linewidth=1.2)
        imag = plt.imshow(arr.T, interpolation='nearest', origin='lower', vmin=-1., vmax=1.)
        plt.colorbar()

else: 
    is_master = True if rank == 1 else False
    getf = gpu.GetFields(gpuf, 'ez', (0, 0, 0), (0, 0, 0))


if is_master:
    #print 'ns', gpuf.ns
    #print 'nbytes (MB)', nx*ny*nz * 9 * 4. / (1024**2)

    from datetime import datetime
    from time import time, sleep
    t0 = datetime.now()
    t00 = time()

# main loop
for tstep in xrange(1, tmax+1):
    nodef.update_e()
    nodef.update_h()

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

if is_master:
    dt = (time() - t00) / tmax
    throughput = size*nx*ny*nz/dt
    print throughput
    #print('[%s] %d/%d (%d %%) %f Mpoint/s' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100, nx*ny*nz*tmax/dt/1e6) )
