#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import *

'''
rc("font", family="serif")
rc("font", size=12)

width = 4.5
height = 1.4
rc("figure.subplot", left=(22/72.27)/width)
rc("figure.subplot", right=(width-10/72.27)/width)
rc("figure.subplot", bottom=(14/72.27)/height)
rc("figure.subplot", top=(height-7/72.27)/height)
fig = figure(figsize=(width, height))
'''

fig = figure(dpi=150)
#ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,9), ylim=(0,45))
ax = fig.add_axes([0.1, 0.15, 0.85, 0.8])	#[left, bottom, width, height] 
ax.set_xlim(0,9)
ax.set_ylim(0,45)
cpu_gflops = [4.58, 5.56]
gpu_gflops = [13.86, 20.77, 35.59, 37.90, 40.57]

width = 0.35
sf = width/2
bar_cpu = ax.bar(np.array([1,2])-sf, cpu_gflops, width, color='w')
bar_gpu = ax.bar(np.array([3,4,5,6,7,])-sf, gpu_gflops, width, color='0.5')

ax.set_xlabel('Intel CPUs and NVIDIA GPUs', fontsize=15)
ax.set_ylabel('Attainable FLOPS (GFLOPS)', fontsize=15)
ax.set_xticks(range(9))
ax.set_xticklabels(['', 'i7 920', 'i7 2600', '9800 GTX+', 'C1060', 'GTX 280', 'C2050', 'GTX 480', ''], rotation=20)

ax.legend( (bar_cpu[0], bar_gpu[0]), ('Intel CPUs', 'NVIDIA GPUs'),  loc='upper left')
show()
