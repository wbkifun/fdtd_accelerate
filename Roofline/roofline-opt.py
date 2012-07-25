#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import *

xmin, xmax = 2**-4, 2**-1
ymin, ymax = 2**1, 2**6

fig = figure(dpi=150)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(xmin,xmax), ylim=(ymin,ymax) )
ax.set_xscale("log", basex=2)
ax.set_yscale("log", basey=2)
ax.set_xticklabels(['1/16','1/8', '1/4', '1/2'])
ax.set_yticklabels(['', '4', '8', '16', '32', '64'])
ax.set_xlabel('Operational Intensity (FLOP/Byte)', fontsize=15)
ax.set_ylabel('Attainable FLOPS (GFLOPS)', fontsize=15)
#ax.grid()

LW = 1		# line width
MSIZE = 10	# marker size
MFC = 'k'	# marker face color

ax.plot([xmin, 1051/129.7, xmax], [xmin*129.7, 1051, 1051], 'k-', lw=2)
ax.plot([0.1667], [21.62], 'kD', markersize=MSIZE, markerfacecolor='w')
ax.plot([0.1786], [23.16], 'kD', ms=MSIZE, mfc='w')
ax.plot([0.2339], [30.34], 'kD', ms=MSIZE, mfc='w')
l_0, = ax.plot([0.1667,0.1667], [1,21.75], 'ko:', lw=LW, ms=MSIZE, mfc=MFC, )
l_1, = ax.plot([0.1786,0.1786], [1,23.08], 'ko--', lw=LW, ms=MSIZE, mfc=MFC)
l_2, = ax.plot([0.2339,0.2339], [1,31.80], 'ko-', lw=LW, ms=MSIZE, mfc=MFC)
#legend((l_0, l_1, l_2), ('Naively implemented', 'Remove mis-aligned', 'Exploit temporal locality'), loc='lower right', numpoints=1)

ax.plot([xmin, 936/73.74, xmax], [xmin*73.74, 936, 936], 'k-', lw=2)
ax.plot([0.1667], [12.29], 'kD', ms=MSIZE, mfc='w')
ax.plot([0.1786], [13.17], 'kD', ms=MSIZE, mfc='w')
ax.plot([0.2339], [17.25], 'kD', ms=MSIZE, mfc='w')
ax.plot([0.1667,0.1667], [1,11.62], 'ko:', lw=LW, ms=MSIZE, mfc=MFC)
ax.plot([0.1786,0.1786], [1,12.41], 'ko--', lw=LW, ms=MSIZE, mfc=MFC)
ax.plot([0.2339,0.2339], [1,17.49], 'ko-', lw=LW, ms=MSIZE, mfc=MFC)

ax.plot([xmin, 705/54.98, xmax], [xmin*54.98, 705, 705], 'k-', lw=2)
ax.plot([0.1071], [5.88], 'kD', ms=MSIZE, mfc='w')
ax.plot([0.1786], [9.82], 'kD', ms=MSIZE, mfc='w')
m0 = ax.plot([0.2339], [12.86], 'kD', ms=MSIZE, mfc='w')
ax.plot([0.1071,0.1071], [1,5.38], 'ko:', lw=LW, ms=MSIZE, mfc=MFC)
ax.plot([0.1786], [9.28], 'ko', lw=LW, ms=MSIZE, mfc=MFC)
m1 = ax.plot([0.2339], [12.16], 'ko', lw=LW, ms=MSIZE, mfc=MFC)
l1 = legend((m0, m1), ('Prediction', 'Experiment'), loc='lower right', numpoints=1)

ax.text(0.07, 0.07*129.7+0.5, 'GeForce GTX 280', fontsize=14, rotation=25)
ax.text(0.07, 0.07*73.74+0.3, 'TESLA C1060', fontsize=14, rotation=25)
ax.text(0.07, 0.07*54.98+0.2, 'GeForce 9800 GTX+', fontsize=14, rotation=25)

#ax.text(0.185, 2.2, 'Remove mis-aligned', color='k', rotation=90)
#ax.text(0.242, 2.2, 'Reduce duplicated', color='k', rotation=90)
ax.plot([0], [0], 'k:', ms=MSIZE, mfc='k', label='Naive')
ax.plot([0], [0], 'k--', ms=MSIZE, mfc='k', label='Remove mis-aligned')
ax.plot([0], [0], 'k-', ms=MSIZE, mfc='k', label='Exploit temporary locality')
l2 = ax.legend(loc='upper left', numpoints=1)
gca().add_artist(l1)

show()
