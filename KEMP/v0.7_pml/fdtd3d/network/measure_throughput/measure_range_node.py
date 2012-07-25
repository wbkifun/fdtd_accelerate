import numpy as np
import subprocess as sp


nodes = ['g101', 'g102', 'g103', 'g104', 'g105', 'ghome']
cmd_base = 'mpirun -host %s python /home/kifang/kemp/fdtd3d/network/measure_throughput/measure_mpi.py'
#cmd_base = 'mpirun -host %s python /home/kifang/kemp/fdtd3d/network/measure_throughput/measure_mpi_split.py'


thp = np.zeros(5)
sub_thp = np.zeros(5)

for i in range(5):
    cmd = cmd_base % ','.join(nodes[:i+2])
    for j in range(5):
        proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        sub_thp[j] = float(stdout)

    thp[i] = sub_thp.mean()

    print i+2, thp[i]


np.save('./throughput_node.npy', thp)
