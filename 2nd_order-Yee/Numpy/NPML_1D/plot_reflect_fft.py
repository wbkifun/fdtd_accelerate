#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import sys

f = h5.File(sys.argv[1], 'r')
ez_ref = f['ez_ref'].value
kez_ref = np.abs( np.fft.fft(ez_ref) )
bin_k = 2*np.pi*np.fft.fftfreq(1024, 1)
bin_wl = 1./np.fft.fftfreq(1024, 1)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.plot(ez_ref)

'''
ax.plot(bin_k, kez_ref, 'o-')
ax.set_xlim(min(bin_k), max(bin_k))
ax.set_xlabel('k')
ax.set_ylabel('Amplitude (A.U.)')
'''

ax.plot(bin_wl, kez_ref, 'o-')
#ax.set_xlim(min(bin_wl), -min(bin_wl))
ax.set_xlim(0, -min(bin_wl))
ax.set_xlabel('Wavelength (dx)')
ax.set_ylabel('Amplitude (A.U.)')

plt.show()
