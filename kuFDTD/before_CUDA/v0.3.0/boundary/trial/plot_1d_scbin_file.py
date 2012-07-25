#!/usr/bin/env python

from scipy.io.numpyio import fread
from kufdtd.util.read_scbin_file import read_scbin_1d

from pylab import *

import sys
import glob
import scipy as sc

#--------------------------------------------------------------------------
# read the parameters
#--------------------------------------------------------------------------
try:
	filepath = sys.argv[1]
except IndexError:
	print 'Error: Wrong arguments'
	print 'Usage: ./plot_1d_scbin_file.py filepath'
	sys.exit()

print 'filepath= ', filepath
#Nx = 400
period = 240
#interval = 4.25*period
#Nx = int( round(3*interval) )
Nx = int( round(3*period) )

#---------------------------------------------
# plot

#fig = figure(figsize=(10,7.5), dpi=150)	# inch
fig = figure(dpi=150, facecolor='white')

grid = read_scbin_1d(filepath, Nx+1)

#line, = plot(grid,'.',ms=20)
line, = plot(grid,'.')
#xticks(arange(950,971))
#axvline(x=959, c='k', ls='--')
#axvline(x=960, c='k', ls='--')

#yticks([0.499945, 0.5, 0.500055])
#axis([240,240*7,0.5001,0.4999])
#axis([240,240*7,0.499955,0.500043])

show()
