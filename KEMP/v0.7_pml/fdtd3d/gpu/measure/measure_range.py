import numpy as np
import subprocess as sp

cmd_base = 'python /home/kifang/kemp/fdtd3d/gpu/measure/measure.py'

point = np.zeros(25, np.int64)
dts = np.zeros(25)
sub_dt = np.zeros(5)

for i in xrange(25):
    nx = (i+1) * 16
    cmd = cmd_base + ' %d' % nx 

    for j in xrange(5):
        proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        sub_dt[j] = float(stdout)

    point[i] = nx * 256 * 256
    dts[i] = sub_dt.mean()

    print i, nx, dts[i], point[i]/dts[i]/1e6

np.save('./measure_point.npy', point)
np.save('./measure_dts.npy', dts)
