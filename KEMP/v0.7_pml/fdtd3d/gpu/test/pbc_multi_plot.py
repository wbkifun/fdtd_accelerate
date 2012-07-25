import numpy as np
import pyopencl as cl

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.gpu import Fields, Core, Pbc, IncidentDirect, GetFields


tmax = 150
tfunc = lambda tstep: np.sin(0.05 * tstep)

# plot
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower')
#plt.ion()
fig = plt.figure(figsize=(14,8))

# gpu device
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]

# z-axis
nx, ny, nz = 180, 160, 2
fields = Fields(context, device, nx, ny, nz)
Core(fields)
Pbc(fields, 'xyz')
IncidentDirect(fields, 'ey', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
IncidentDirect(fields, 'ex', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

ax1 = fig.add_subplot(2, 3, 1)
getf = GetFields(fields, 'ey', (0, 0, nz/2), (nx-1, ny-1, nz/2))
getf.get_event().wait()
ax1.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax1.set_title('%s, ey[20,:,:]' % repr(fields.ns))
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(2, 3, 4)
getf = GetFields(fields, 'ex', (0, 0, nz/2), (nx-1, ny-1, nz/2))
getf.get_event().wait()
ax2.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax2.set_title('%s, ex[:,20,:]' % repr(fields.ns))
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# y-axis 
nx, ny, nz = 180, 2, 160
fields = Fields(context, device, nx, ny, nz)
Core(fields)
Pbc(fields, 'xyz')
IncidentDirect(fields, 'ez', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
IncidentDirect(fields, 'ex', (0, 0, 20), (nx-1, ny-1, 20), tfunc) 

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

ax1 = fig.add_subplot(2, 3, 2)
getf = GetFields(fields, 'ez', (0, ny/2, 0), (nx-1, ny/2, nz-1))
getf.get_event().wait()
ax1.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax1.set_title('%s, ez[20,:,:]' % repr(fields.ns))
ax1.set_xlabel('x')
ax1.set_ylabel('z')

ax2 = fig.add_subplot(2, 3, 5)
getf = GetFields(fields, 'ex', (0, ny/2, 0), (nx-1, ny/2, nz-1))
getf.get_event().wait()
ax2.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax2.set_title('%s, ex[:,:,20]' % repr(fields.ns))
ax2.set_xlabel('x')
ax2.set_ylabel('z')

# x-axis 
nx, ny, nz = 2, 180, 160
fields = Fields(context, device, nx, ny, nz)
Core(fields)
Pbc(fields, 'xyz')
IncidentDirect(fields, 'ez', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
IncidentDirect(fields, 'ey', (0, 0, 20), (nx-1, ny-1, 20), tfunc) 

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

ax1 = fig.add_subplot(2, 3, 3)
getf = GetFields(fields, 'ez', (nx/2, 0, 0), (nx/2, ny-1, nz-1))
getf.get_event().wait()
ax1.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax1.set_title('%s, ez[:,20,:]' % repr(fields.ns))
ax1.set_xlabel('y')
ax1.set_ylabel('z')

ax2 = fig.add_subplot(2, 3, 6)
getf = GetFields(fields, 'ey', (nx/2, 0, 0), (nx/2, ny-1, nz-1))
getf.get_event().wait()
ax2.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax2.set_title('%s, ey[:,:,20]' % repr(fields.ns))
ax2.set_xlabel('y')
ax2.set_ylabel('z')


#plt.savefig('./png/%.6d.png' % tstep)
plt.show()
