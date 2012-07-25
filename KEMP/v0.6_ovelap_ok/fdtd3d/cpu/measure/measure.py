import numpy as np

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.cpu import QueueTask, Fields, Core, Pbc, IncidentDirect, GetFields


try:
    ny = int(sys.argv[1])
except:
    ny = 256

nx, nz = 2, 256
tmax, tgap = 1000, 10 

# instances 
fields = Fields(QueueTask(), nx, ny, nz, coeff_use='e', precision_float='single', use_cpu_core=1)
Core(fields)
fields2 = Fields(QueueTask(), nx, ny, nz, coeff_use='e', precision_float='single', use_cpu_core=1)
Core(fields2)

#print 'ns_pitch', fields.ns_pitch
#print 'nbytes (MB)', nx*ny*nz * 9 * 4. / (1024**2)

'''
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
'''

# main loop
from datetime import datetime
from time import time
t0 = datetime.now()
t00 = time()

gtmp = GetFields(fields, 'ez', (0, 0, 0), (0, 0, 0))
gtmp2 = GetFields(fields2, 'ez', (0, 0, 0), (0, 0, 0))
for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields2.update_e()
    fields.update_h()
    fields2.update_h()

    '''
    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.get_event().wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()
    '''

#plt.show()
gtmp.get_event().wait()
gtmp2.get_event().wait()
dt = time() - t00
print dt/tmax
#print('[%s] %d/%d (%d %%) %f Mpoint/s' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100, nx*ny*nz*tmax/dt/1e6) )
