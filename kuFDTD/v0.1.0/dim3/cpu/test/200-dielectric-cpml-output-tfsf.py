#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric
from kufdtd.dim3.cpu.cpml import CpmlNonKappa
from kufdtd.dim3.output import Output
from kufdtd.dim3.tfsf import Tfsf


#--------------------------------------------------------------------
Nx, Ny, Nz = 200, 200, 64
dx = 10e-9
tmax = 1000
Ncore = 8

Npml = 15
wavelength = 600e-9

Cpml = CpmlNonKappa( Npml, ('fb','fb','fb') )

Output_ey = Output( 'Ey', (None, None, Nz/2), (None, None, Nz/2) ) 
output_list = [Output_ey]

Src = Tfsf( (50, 50, 20), (150, 150, 40), ('fb','fb','fb'), wavelength, ['normal', 'x'], 0 )
src_list = [Src]

#--------------------------------------------------------------------
S = Dielectric( Nx, Ny, Nz, dx, Ncore )

S.allocate_main()
S.allocate_coeff()
S.set_coeff()

Cpml.set_space( S )
Cpml.allocate_psi()
Cpml.allocate_coeff()

for output in output_list: output.set_space( S )
for src in src_list: src.set_space( S )

#--------------------------------------------------------------------
print '-'*47
print 'N(%d, %d, %d)' % (S.Nx, S.Ny, S.Nz)
print 'dx = %g' % S.dx
print 'dt = %g' % S.dt
print 'Npml = %g' % Cpml.Npml
print ''
S.print_memory_usage()
#Cpml.print_memory_usage()
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
	Cpml.updateE()
	Src.updateE( tstep )

	S.updateH()
	Cpml.updateH()
	Src.updateH()
	
	if tstep/20*20 == tstep:
		print_elapsed_time( t0, time(), tstep )
		
		imsh.set_array( transpose( Output_ey.get_data() ) )
		png_str = './png/%.6d.png' % tstep
		savefig(png_str) 
		

print_elapsed_time( t0, time(), tstep )
