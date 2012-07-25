import numpy as np
import pyopencl as cl

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.node import Fields, Core, IncidentDirect, GetFields, Pbc
from kemp.fdtd3d import gpu, cpu


ny, nz = 140, 2
gpu_nx = 141
cpu_nx = 20
tmax, tgap = 150, 10 

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
mainf_list = [ cpu.Fields(cpu_nx, ny, nz) ]
mainf_list += [gpu.Fields(context, device, gpu_nx, ny, nz) for device in gpu_devices]

fields = Fields(mainf_list)
Core(fields)
Pbc(fields, 'xyz')
nx = fields.nx

tfunc = lambda tstep: np.sin(0.05 * tstep)
#IncidentDirect(fields, 'ez', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
IncidentDirect(fields, 'ez', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
getf = GetFields(fields, 'ez', (0, 0, nz/2), (nx-1, ny-1, nz/2))

#IncidentDirect(fields, 'ey', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
#getf = GetFields(fields, 'ey', (0, 0, nz/2), (nx-1, ny-1, nz/2))

#IncidentDirect(fields, 'ex', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
#getf = GetFields(fields, 'ex', (0, 0, nz/2), (nx-1, ny-1, nz/2))

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
for i in fields.accum_nx_list[1:]:
	plt.plot((i,i), (0,ny), color='k', linewidth=2)
imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.colorbar()

# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
