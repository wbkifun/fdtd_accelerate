import numpy as np
import pyopencl as cl

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.node import NodeFields, NodeCore, NodeExchange, NodeDirectIncident, NodeGetFields, NodePbc
from kemp.fdtd3d import gpu, cpu


nx_gpu = 120
nx_cpu = 80
#nx_gpu = nx_cpu = 100
ny, nz = 300, 64
tmax, tgap = 200, 10

# instances
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)

mainf_list = [gpu.Fields(context, device, nx_gpu, ny, nz) for device in gpu_devices]
mainf_list.append( cpu.Fields(nx_cpu, ny, nz) )
nodef = NodeFields(mainf_list)
core = NodeCore(nodef)
exchange = NodeExchange(nodef)
pbc = NodePbc(nodef, 'y')
pbc = NodePbc(nodef, 'z')
pbc_x = NodePbc(nodef, 'x')

tfunc = lambda tstep: np.sin(0.1 * tstep)
#incident = NodeDirectIncident(nodef, 'ez', (0, 20, 0), (nodef.nx-1, 20, nz-1), tfunc) 
incident = NodeDirectIncident(nodef, 'ez', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
getf = NodeGetFields(nodef, 'ez', (0, 0, 2), (nodef.nx-1, ny-1, 2))

# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.ion()
for anx in nodef.accum_nx_list[1:]:
	plt.plot((anx, anx), (0, ny), color='w', linewidth=0.2)
imag = plt.imshow(np.zeros((nodef.nx, ny), nodef.dtype).T, cmap=plt.cm.hot, origin='lower', vmin=0, vmax=2.)
plt.colorbar()

# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    nodef.update_e()
    pbc_x.update_e()
    exchange.update_e()

    nodef.update_h()
    pbc_x.update_h()
    exchange.update_h()

    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.wait()
        imag.set_array( np.abs(getf.get_fields().T) )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
#getf.wait()
#print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
