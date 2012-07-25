import numpy as np
import subprocess as sp


#cmd_base = 'mpirun -host g101,g102,g103,g104,g105 python /home/kifang/kemp/fdtd3d/network/measure_throughput/measure_mpi_split.py'
cmd_base = 'mpirun -host g101,g102,g103,g104,g105 python /home/kifang/kemp/fdtd3d/network/measure_throughput/measure_mpi.py'


nxs = range(200, 400, 5)
thp = np.zeros(len(nxs), np.int64)
sub_thp = np.zeros(5)

for i, nx in enumerate(nxs):
    cmd = cmd_base + ' %d' % nx
    '''
    for j in xrange(5):
        proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        sub_thp[j] = float(stdout)
    thp[i] = sub_thp.mean()
    '''
    proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    thp[i] = float(stdout)

    print i, nx, thp[i]


#np.save('./measure_nx_split.npy', nxs)
#np.save('./measure_throughput_split.npy', thp)
np.save('./measure_nx_buffer.npy', nxs)
np.save('./measure_throughput_buffer.npy', thp)
