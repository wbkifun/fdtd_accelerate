import numpy as np

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.gpu import Fields, Core, DirectIncident, GetFields, Pbc
from kemp.fdtd3d.util import common_exchange


nx, ny, nz = 200, 300, 16
tmax, tgap = 200, 10

# instances
from kemp.fdtd3d.util import common_gpu
import pyopencl as cl
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
fields = Fields(context, gpu_devices[0], nx, ny, nz)
#fields = Fields(nx, ny, nz)

core = Core(fields)
pbc = Pbc(fields, 'x')
pbc = Pbc(fields, 'y')
pbc = Pbc(fields, 'z')
print fields.instance_list

tfunc = lambda tstep: np.sin(0.05 * tstep)
incident = DirectIncident(fields, 'ez', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
incident = DirectIncident(fields, 'ey', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
#incident = DirectIncident(fields, 'ex', (0, 0, 20), (nx-1, ny-1, 20), tfunc) 
getf = GetFields(fields, 'ez', (0, 0, nz/2), (nx-1, ny-1, nz/2))

# for verify pbc
vpbc = common_exchange.VerifyPbc(fields, 'xyz')

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
#fig.add_subplot(1, 3, 1)
#imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=0.05)
imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=0, vmax=1.1)
plt.colorbar()

# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    fields.update_e()
    vpbc.verify_e()

    fields.update_h()
    vpbc.verify_h()

    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.get_event().wait()
        imag.set_array( np.abs(getf.get_fields().T) )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
#getf.wait()
#print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
