#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys


rr = np.load(sys.argv[1])
print(rr.shape)
print(rr.min())

plt.ion()
fig = plt.figure(figsize=(16,4))
#plt.imshow(rr.T, origin='lower')
#plt.colorbar()
#plt.show()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.set_xlabel('sigma_h')
ax2.set_xlabel('sigma_h')
ax3.set_xlabel('sigma_h')
ax1.set_ylabel('Reflectance (%)')
ax3.set_title('The lowest R')
lowest_r = 1

xbin = np.arange(0, 5, 0.1)
l1, = ax1.plot(xbin, rr[1,:], 'o-')
l2, = ax2.plot(xbin, rr[1,:], 'o-')
l3, = ax3.plot(xbin, rr[1,:], 'o-')

ax1.set_ylim(0, 100)
ax2.set_ylim(0.2, 1.0)
ax3.set_ylim(0, 100)

for i in xrange(rr.shape[0]):
	l1.set_ydata(rr[i,:])
	l2.set_ydata(rr[i,:])
	ax1.set_title('sigma_e = %g' % (i*0.1))

	min_r = rr[i,:].min()
	ax2.set_title('min(R) = %g' % min_r)

	if min_r < lowest_r:
		print(min_r, lowest_r, i*0.1, rr[i,:].argmin()*0.1)
		l3.set_ydata(rr[i,:])
		lowest_r = min_r

	l1.recache()
	plt.draw()
