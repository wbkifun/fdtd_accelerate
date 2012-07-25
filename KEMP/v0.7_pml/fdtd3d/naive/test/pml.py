import numpy as np

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.naive import Fields, Core, Pbc, IncidentDirect, Pml


nx, ny, nz = 250, 300, 4
tmax, tgap = 300, 20 

# instances 
fields = Fields(nx, ny, nz, segment_nbytes=16)
Core(fields)
Pbc(fields, 'yz')
Pml(fields, ('+', '', ''))

tfunc = lambda tstep: 50 * np.sin(0.05 * tstep)
#IncidentDirect(fields, 'ez', (120, 0, 0), (120, ny-1, nz-1), tfunc) 
#IncidentDirect(fields, 'ez', (0, 20, 0), (nx-1, 20, nz-1), tfunc) 
IncidentDirect(fields, 'ez', (150, ny/2, 0), (150, ny/2, nz-1), tfunc) 

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ioff()
fig = plt.figure(figsize=(12,8))
imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.colorbar()

from matplotlib.patches import Rectangle
npml = 50
rect = Rectangle((nx-npml, 0), npml, ny, alpha=0.1)
plt.gca().add_patch(rect)

# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    #print 'ez', fields.ez[0,:,0]
    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        imag.set_array( fields.ez[:,:,nz/2].T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
