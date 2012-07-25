#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric
from kufdtd.dim3.cpu.cpml import CpmlNonKappa
from kufdtd.dim3.output import Output


#--------------------------------------------------------------------
Nx, Ny, Nz = 200, 200, 32
dx = 10e-9
tmax = 1000
Ncore = 8

Npml = 15

Cpml = CpmlNonKappa( Npml, ('fb','fb','fb') )

Output_ez = Output( 'Ez', (0,0,Nz/2), (Nx-1, Ny-1, Nz/2) ) 
Output_e = Output( 'e', (None,0,Nz/2), (None, Ny-2, Nz/2), (2,2,1) ) 
output_list = [Output_ez, Output_e]

#--------------------------------------------------------------------
S = Dielectric( Nx, Ny, Nz, dx, Ncore )

S.allocate_main()
S.allocate_coeff()
S.set_coeff()

Cpml.set_space( S )
Cpml.allocate_psi()
Cpml.allocate_coeff()

for output in output_list: output.set_space( S )

#--------------------------------------------------------------------
print '-'*47
print 'N(%d, %d, %d)' % (S.Nx, S.Ny, S.Nz)
print 'dx = %g' % S.dx
print 'dt = %g' % S.dt
print 'Npml = %g' % Cpml.Npml
print ''
S.print_memory_usage()
Cpml.print_memory_usage()
print '-'*47

#--------------------------------------------------------------------
# Graphic
from pylab import *
ion()
figure()

imsh = imshow( 
		transpose( sc.ones( (S.Nx,S.Ny), 'f' )[::2,::2] ),
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
	Cpml.updateE()

	S.Ez[Nx/2+20, Ny/2+50, :] += sc.sin(0.1*tstep)

	S.updateH()
	Cpml.updateH()

	
	if tstep/20*20 == tstep:
		print_elapsed_time( t0, time(), tstep )
		
		#imsh.set_array( transpose( S.Ez[:,:,Nz/2] ) )
		#imsh.set_array( transpose( Output_ez.get_data() ) )
		imsh.set_array( transpose( Output_e.get_data() ) )
		png_str = './png/%.6d.png' % tstep
		savefig(png_str) 
		

print_elapsed_time( t0, time(), tstep )
