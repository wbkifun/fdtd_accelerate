import numpy as np
import pyopencl as cl

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.node import Fields, Core, IncidentDirect, GetFields
from kemp.fdtd3d import gpu, cpu

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ny, nz = 140, 32
gpu_nx = 141
cpu_nx = 20
tmax, tgap = 150, 10 

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
mainf_list = [gpu.Fields(context, device, gpu_nx, ny, nz) for device in gpu_devices]
mainf_list.append( cpu.Fields(cpu_nx, ny, nz) )
#mainf_list = [ cpu.Fields(160, ny, nz) ]

fields = Fields(mainf_list, pbc='yz', mpi_shape=(2,1,1), tmax=tmax)
Core(fields)
nx = fields.nx

tfunc = lambda tstep: np.sin(0.05 * tstep)
#IncidentDirect(fields, 'ez', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
IncidentDirect(fields, 'ez', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
getf = GetFields(fields, 'ez', (0, 0, nz/2), (nx-1, ny-1, nz/2))
if rank == 0:
    buf = fields.buffer_dict['x+']
    print 'buf instance_list', buf.instance_list
    getf_buf = cpu.GetFields(buf, 'ez', (0, 0, nz/2), (2, ny-1, nz/2))

#IncidentDirect(fields, 'ey', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
#getf = GetFields(fields, 'ey', (0, 0, nz/2), (nx-1, ny-1, nz/2))

#IncidentDirect(fields, 'ex', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
#getf = GetFields(fields, 'ex', (0, 0, nz/2), (nx-1, ny-1, nz/2))

#print fields.updatef_list
"""
# plot
if rank == 0:
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure(figsize=(12,8))
    '''
    for i in fields.accum_nx_list[1:]:
        plt.plot((i,i), (0,ny), color='k', linewidth=2)
    imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
    '''
    imag = plt.imshow(np.zeros((buf.nx, buf.ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
    plt.colorbar()

# main loop
if rank == 0:
    from datetime import datetime
    t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    print 'rank, tstep', rank, tstep
    fields.update_e()
    fields.update_h()

    if tstep % tgap == 0 and rank == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        #getf.wait()
        #imag.set_array( getf.get_fields().T )
        getf_buf.get_event().wait()
        imag.set_array( getf_buf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

if rank == 0:
    #plt.show()
    print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
    print('')
"""
