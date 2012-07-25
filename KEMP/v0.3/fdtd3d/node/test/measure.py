import numpy as np
import pyopencl as cl
from mpi4py import MPI

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.node import Fields, BufferFields, Core, Pbc
from kemp.fdtd3d import gpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#nx, ny, nz = 240, 256, 256  # 540 MB
#nx, ny, nz = 544, 544, 544  # 5527 MB
#nx, ny, nz = 512, 512, 512  # 4608 MB
#nx, ny, nz = 480, 480, 480  # 3796 MB
nx, ny, nz = 240, 256, 256  # 576 MB
#nx, ny, nz = 128, 128, 128  # 72 MB
tmax, tgap = 150, 10 

coeff_use = 'e'
precision_float = 'single'

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]
gpuf = gpu.Fields(context, device, nx, ny, nz, coeff_use, precision_float)
buffer_dict = {}
if rank == 0: 
    buffer_dict['x+'] = BufferFields(gpuf, 'x+', tmax, ny, nz, coeff_use, precision_float)
elif rank == 1: 
    buffer_dict['x-'] = BufferFields(gpuf, 'x-', tmax, ny, nz, coeff_use, precision_float)

fields = Fields([gpuf], buffer_dict)
Core(fields)
Pbc(fields, 'yz')

if rank == 0:
    print 'ns', fields.ns
    print 'nbytes (MB)', nx*ny*nz * 9 * 4. / (1024**2)

    tfunc = lambda tstep: 40 * np.sin(0.05 * tstep)
    gpu.IncidentDirect(gpuf, 'ez', (220, 20, 0), (220, 20, -1), tfunc) 
    getf = gpu.GetFields(gpuf, 'ez', (0, 0, 0.5), (-1, -1, 0.5))

    # plot
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure(figsize=(12,8))
    imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
    plt.colorbar()

    from datetime import datetime
    from time import time
    t0 = datetime.now()
    t00 = time()

# main loop
gtmp = gpu.GetFields(gpuf, 'ez', (0, 0, 0), (0, 0, 0))
for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    if tstep % tgap == 0 and rank == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.get_event().wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

gtmp.get_event().wait()
if rank == 0:
    dt = time() - t00
    print dt
    print('[%s] %d/%d (%d %%) %f Mpoint/s' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100, nx*ny*nz*tmax/dt/1e6) )

elif rank == 1:
    print 'rank', rank
