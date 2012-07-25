from __future__ import division
import numpy as np


nbyte = 4
ny, nz = 250, 256
gs_comm = ny * nz

thp_gpu =   1188020000
thp_cpu =    109304000 * 0.35

#bw_memcpy = 2028870000 
#lt_memcpy = 0.0000673
bw_memcpy = 1690720000 
lt_memcpy = 0.00008076

#bw_mpi = bw_mpi_buf = 180298000
#lt_mpi =  lt_mpi_buf = 0.000677936
bw_mpi = bw_mpi_buf = 193869000
lt_mpi =  lt_mpi_buf = 0.00063048


t_memcpy = 4 * (2 * nbyte * gs_comm / bw_memcpy + lt_memcpy)
t_mpi = 4 * (2 * nbyte * gs_comm / bw_mpi + lt_mpi)

r0_split = 1 + (thp_gpu / gs_comm) * t_mpi
r0_buf = thp_gpu * (2 / thp_cpu + (t_mpi - t_memcpy) / gs_comm)

print 'r0', r0_split, r0_buf

nx0, nx1 = 200, 400

nxs = np.arange(nx0, nx1, 0.1)
thp_total_buf = np.zeros(nxs.size)
thp_total_split = np.zeros(nxs.size)
rs = np.zeros(nxs.size)

for i, nx in enumerate(nxs):
    gs_gpu = nx * ny * nz
    r = float(gs_gpu) / gs_comm
    rs[i] = r

    if r >= r0_split:
        thp_total_split[i] = 1 / (1 / thp_gpu + 1/r * t_memcpy / gs_comm) 

    if r >= r0_buf:
        thp_total_buf[i] = 1 / (1 / thp_gpu + 1/r * t_memcpy / gs_comm) 


    if r < r0_split:
        thp_total_split[i] = r / (1 / thp_gpu + (t_mpi + t_memcpy) / gs_comm)

    if r < r0_buf:
        thp_total_buf[i] = r / (2 / thp_cpu + t_mpi / gs_comm)


thp_total_split /= thp_gpu
thp_total_buf /= thp_gpu

n_node = 5
exp_nxs = np.load('npy/measure_nx_buffer5_2.npy')
exp_thp_split = np.load('npy/measure_throughput_split5_2.npy') / n_node / thp_gpu
exp_thp_buf = np.load('npy/measure_throughput_buffer5_2.npy') / n_node / thp_gpu

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.ion()
fig = plt.figure(dpi=150)
ax1 = fig.add_subplot(1,1,1)

#p0 = ax1.plot([rs[0], rs[-1]], [thp_gpu*n_node, thp_gpu*n_node], 'k-.', linewidth=2)
p0 = ax1.plot([rs[0], rs[-1]], [1, 1], 'k-.', linewidth=2)
p1 = ax1.plot(rs, thp_total_split, 'k--')
p2 = ax1.plot(rs, thp_total_buf, 'k-')
p3 = ax1.plot(exp_nxs, exp_thp_split, linestyle='None', color='w', marker='s')
p4 = ax1.plot(exp_nxs, exp_thp_buf, linestyle='None', color='k', marker='o')
ax1.set_xlabel('Ratio of grid size (computation / communication)')
ax1.set_ylabel('Throughput [A.U.]')
ax1.legend((p0, p1, p2, p3, p4), ('Upper limit', 'Prediction (kernel-split)', 'Prediction (host-buffer)', 'Experiment (kernel-split)', 'Experiment (host-buffer)'), loc='lower right', numpoints=1)
ax1.set_ylim(0.60, 1.05)
#plt.savefig('prediction_experiment.eps', dpi=150)
plt.show()

