#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric, DrudeAde
from kufdtd.dim3.cpu.cpml import CpmlNonKappa


#--------------------------------------------------------------------
Nx, Ny, Nz = 200, 200, 32
dx = 10e-9
tmax = 1000
Ncore = 8

Npml = 15

Cpml = CpmlNonKappa( Npml, ('fb','fb','fb') )

#--------------------------------------------------------------------
S = Dielectric( Nx, Ny, Nz, dx, Ncore )

S.allocate_main()
S.allocate_coeff()

Drude = DrudeAde( Nx, Ny, Nz, 0, 0, 0, S )
Drude.allocate_main()
Drude.allocate_coeff()

epr = 9.0685
pfreq = 2*pi*2155.6*1e12/sc.sqrt(epr)
gamma = 2*pi*18.36*1e12

S.epr_x[50:101,50:100,:] = epr
S.epr_y[50:100,50:101,:] = epr
S.epr_z[50:100,50:100,:] = epr
S.set_coeff()

Drude.pfreq_x[50:101,50:100,:] = pfreq
Drude.pfreq_y[50:100,50:101,:] = pfreq
Drude.pfreq_z[50:100,50:100,:] = pfreq
Drude.gamma_x[50:101,50:100,:] = gamma
Drude.gamma_y[50:100,50:101,:] = gamma
Drude.gamma_z[50:100,50:100,:] = gamma
Drude.set_coeff()

Cpml.set_space( S )
Cpml.allocate_psi()
Cpml.allocate_coeff()

#--------------------------------------------------------------------
print '-'*47
print 'N(%d, %d, %d)' % (S.Nx, S.Ny, S.Nz)
print 'dx = %g' % S.dx
print 'dt = %g' % S.dt
print 'Npml = %g' % Cpml.Npml
print ''
S.print_memory_usage()
Drude.print_memory_usage()
Cpml.print_memory_usage()
print '-'*47

#--------------------------------------------------------------------
# Graphic
from pylab import *
ion()
figure()

imsh = imshow( 
			transpose( sc.ones( (S.Nx,S.Ny), 'f' ) ),
			cmap=cm.jet,
			vmin=-0.05, vmax=0.05,
			origin='lower',
			interpolation='bilinear')
colorbar()


#--------------------------------------------------------------------
from time import *
t0 = time()
for tstep in xrange( 1, tmax+1 ):
	S.updateE()
	Drude.updateE()
	Cpml.updateE()

	S.Ez[Nx/2+30, Ny/2+50, :] += sc.sin(0.1*tstep)

	S.updateH()
	Cpml.updateH()

	
	if tstep/20*20 == tstep:
		print_elapsed_time( t0, time(), tstep )
		
		imsh.set_array( transpose( S.Ez[:,:,Nz/2] ) )
		png_str = './png/%.6d.png' % tstep
		savefig(png_str) 
		

print_elapsed_time( t0, time(), tstep )
