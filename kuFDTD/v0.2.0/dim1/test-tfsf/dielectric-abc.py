'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 7. 16
 last update  : 

 Copyright : GNU GPL
'''

import scipy as sc
from pylab import *

light_velocity = 299792458
Nx = 200
dx = 10e-9
dt = dx/(2*light_velocity)
wavelength = 600e-9
wfreq = 2*sc.pi*light_velocity/wavelength

E = sc.zeros( Nx, 'f' )
H = sc.zeros( Nx, 'f' )
abc_f1, abc_f2, abc_b1, abc_b2 = 0, 0, 0, 0

# for graphic
ion()
figure()

line, = plot( sc.ones( Nx, 'f') )
axis( [0, Nx, -1.2, 1.2] )

for tstep in xrange( 1, 501 ):
	E[1:-1] -= 0.5*( H[2:] - H[1:-1] ) 

	# abc
	E[0] = abc_f1
	abc_f1 = abc_f2
	abc_f2 = E[1]

	E[-1] = abc_b1
	abc_b1 = abc_b2
	abc_b2 = E[-2]

	E[50] = sc.sin( wfreq*dt*tstep )

	H[1:] -= 0.5*( E[1:] - E[:-1] )

	if tstep/5*5 == tstep:
		line.set_ydata( E )
		draw()
