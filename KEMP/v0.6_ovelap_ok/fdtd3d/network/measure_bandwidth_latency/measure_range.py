import numpy as np
import subprocess as sp


cmd_base = 'mpirun -host g101,g102,g103,g104,g105 python /home/kifang/kemp/fdtd3d/network/measure_bandwidth_latency/measure_mpi.py'


nbytes = np.zeros(25, np.int64)
dts = np.zeros(25)
sub_dt = np.zeros(5)

for i in xrange(25):
    ny = (256 + i * 16)
    cmd = cmd_base + ' %d' % ny
    '''
    for j in xrange(5):
        proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        sub_dt[j] = float(stdout)

    nbytes[i] = ny * 256 * 4 * 4    # ny, nz, fields, nbyte(single)
    dts[i] = sub_dt.mean()
    '''
    proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    dts[i] = float(stdout) / 4      # number of communications
    nbytes[i] = ny * 256 * 2 * 4     # ny, nz, fields, nbyte(single)

    print i, ny, dts[i], nbytes[i]/dts[i]/(1000**2)


np.save('./measure_nbytes.npy', nbytes)
np.save('./measure_dts.npy', dts)
