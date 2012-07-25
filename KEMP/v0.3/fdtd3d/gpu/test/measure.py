import numpy as np
import pyopencl as cl

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.gpu import Fields, Core, Pbc, IncidentDirect, GetFields


#nx, ny, nz = 240, 256, 256  # 540 MB
#nx, ny, nz = 544, 544, 544  # 5527 MB
#nx, ny, nz = 512, 512, 512  # 4608 MB
#nx, ny, nz = 480, 480, 480  # 3796 MB
nx, ny, nz = 800, 256, 256  # 576 MB
#nx, ny, nz = 128, 128, 128  # 72 MB
tmax, tgap = 1000, 10 

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]
fields = Fields(context, device, nx, ny, nz, coeff_use='e', precision_float='single')
Core(fields)

print 'ns_pitch', fields.ns_pitch
print 'nbytes (MB)', nx*ny*nz * 9 * 4. / (1024**2)

'''
Pbc(fields, 'xyz')

tfunc = lambda tstep: 40 * np.sin(0.05 * tstep)
IncidentDirect(fields, 'ez', (220, 20, 0), (220, 20, -1), tfunc) 
getf = GetFields(fields, 'ez', (0, 0, 0.5), (-1, -1, 0.5))

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.colorbar()
'''

# main loop
from datetime import datetime
from time import time
t0 = datetime.now()
t00 = time()

gtmp = GetFields(fields, 'ez', (0, 0, 0), (0, 0, 0))
for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    '''
    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.get_event().wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()
    '''

gtmp.get_event().wait()
dt = time() - t00
print dt
print('[%s] %d/%d (%d %%) %f Mpoint/s' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100, nx*ny*nz*tmax/dt/1e6) )
