import numpy as np

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.cpu import Fields, Core, Pbc, IncidentDirect, GetFields


nx, ny, nz = 160, 140, 32
tmax, tgap = 150, 10 

# instances 
fields = Fields(nx, ny, nz)
Core(fields)
Pbc(fields, 'xyz')

tfunc = lambda tstep: np.sin(0.05 * tstep)
IncidentDirect(fields, 'ez', (20, 0, 0), (20, ny-1, nz-1), tfunc) 
#IncidentDirect(fields, 'ez', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
getf = GetFields(fields, 'ez', (0, 0, nz/2), (nx-1, ny-1, nz/2))

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
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

        getf.get_event().wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
