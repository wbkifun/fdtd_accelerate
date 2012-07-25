#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric


#--------------------------------------------------------------------
Nx, Ny, Nz = 200, 200, 200
dx = 10e-9
tmax = 500
Ncore = 4

#--------------------------------------------------------------------
S = Dielectric( Nx, Ny, Nz, dx, Ncore )

S.allocate_main()
S.allocate_coeff()
S.set_coeff()

#--------------------------------------------------------------------
print '-'*47
print 'N(%d, %d, %d)' % (S.Nx, S.Ny, S.Nz)
print 'dx = %g' % S.dx
print 'dt = %g' % S.dt
print 'tmax = %d' % tmax
print 'Ncore = %d' % Ncore
print ''
print 'memory usage:'
#print_mem_usage( [S.mem_usage] )
print '-'*47

'''
#--------------------------------------------------------------------
# Graphic
from pylab import *
ion()
figure()

imsh = imshow( transpose( sc.ones( (S.Nx,S.Ny), 'f' ) ),
				cmap=cm.jet,
				vmin=-0.05, vmax=0.05,
				origin='lower',
				interpolation='bilinear')
colorbar()
'''

#--------------------------------------------------------------------
from time import *
t0 = time()
for tstep in xrange( 1, tmax+1 ):
	S.updateE()

	#S.Ez[Nx/2-30, Ny/2-50, :] += sc.sin(0.1*tstep)

	#S.updateH()

	if tstep/50*50 == tstep:
		print_elapsed_time( t0, time(), tstep )
		
		'''
		imsh.set_array( transpose( S.Ez[:,:,Nz/2] ) )
		png_str = './png/Ez-%.6d.png' % tstep
		savefig(png_str) 
		'''
	

print_elapsed_time( t0, time(), tstep )
