from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.ion()
fig = plt.figure(dpi=150)
ax1 = fig.add_subplot(1,1,1)

linear = [1.18802, 2.37604, 3.56406, 4.75208, 5.94010, 7.12812]
'''
# 400 x 250 x 256
split = [1.1880, 2.34335, 3.45206, 4.56159, 5.67916, 6.84704]
buf = [1.1880, 2.36204, 3.49597, 4.53066, 5.60487, 6.71079]
nonblock = [1.1880, 1.63912, 2.11546, 2.71490, 3.53842, 4.22414] 
block = [1.1880, 1.63832, 1.87936, 2.04866, 2.16190, 2.24752] 
'''

thp_gpu = 1188020000 / 1e9
linear = [thp_gpu * i for i in range(1,7)]

# 360 x 250 x 256   # 791 MB
#split = [thp_gpu] + list( np.load('npy/throughput_node_split2.npy') / 1e9 )
#buf = [thp_gpu] + list( np.load('npy/throughput_node_buffer3.npy') / 1e9 )
#nonblock = [thp_gpu] + list( np.load('npy/throughput_node_nonblock3.npy') / 1e9 )
#block = [thp_gpu] + list( np.load('npy/throughput_node_block3.npy') / 1e9 )
split = np.load('npy/throughput_node_split2.npy') / 1e9
buf = np.load('npy/throughput_node_buffer3.npy') / 1e9
nonblock = np.load('npy/throughput_node_nonblock3.npy') / 1e9
block = np.load('npy/throughput_node_block3.npy') / 1e9
print 'split', split
print 'buf', buf


xticks = range(1, 7)
xticks2 = range(2, 7)
p0 = ax1.plot(xticks, linear, 'ko-.', linewidth=2)
p1 = ax1.plot(xticks2, split, 'ko-', markerfacecolor='w')
p2 = ax1.plot(xticks2, buf, 'kx-')
p3 = ax1.plot(xticks2, nonblock, 'ks--')
p4 = ax1.plot(xticks2, block, 'k^--')

ax1.set_xlabel('Number of MPI nodes')
ax1.set_ylabel('Throughput [Gpoint/s]')
ax1.set_xlim(0.8, 6.2)
ax1.set_ylim(0, 7.5)
ax1.legend((p0, p1, p2, p3, p4), ('Upper limit', 'Kernel-split', 'Host-buffer', 'MPI nonblocking', 'MPI blocking'), loc='upper left', numpoints=1)

plt.savefig('comparison.eps', dpi=150)
plt.savefig('comparison.png', dpi=150)
#plt.show()
