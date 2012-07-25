#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.gpu.base import *
from kufdtd.dim3.gpu.matter import Dielectric
from kufdtd.dim3.gpu.cpml import CpmlNonKapa
from kufdtd.dim3.gpu.source import Source

import pycuda.autoinit


#--------------------------------------------------------------------
Nx, Ny, Nz = 400, 400, 400
dx = 10e-9
tmax = 1000

#--------------------------------------------------------------------
S = Dielectric( Nx, Ny, Nz, dx )

S.allocate_main_in_dev()
S.initmem_main_in_dev()
S.allocate_coeff_in_dev()

S.allocate_coeff()
S.set_coeff()
S.memcpy_htod_coeff()

S.prepare_kernels()

#--------------------------------------------------------------------
Src = Source( S )
Src.prepare_kernels()

#--------------------------------------------------------------------
Npml = 15
pml_direction = ( 'fb', 'fb', 'fb' )

Cpml = CpmlNonKapa( Npml, S )
Cpml.allocate_psi_in_dev()
Cpml.initmem_psi_in_dev()

Cpml.allocate_coeff()
Cpml.set_coeff()
Cpml.get_module()
Cpml.memcpy_htod_coeff()

Cpml.prepare_kernels()

#--------------------------------------------------------------------
print '-'*47
print 'N(%d, %d, %d)' % (S.Nx, S.Ny, S.Nz)
print 'dx = %g' % S.dx
print 'dt = %g' % S.dt
print 'Npml = %g' % Cpml.Npml
print ''
S.print_kernel_parameters()
S.print_memory_usage()
Cpml.print_kernel_parameters()
Cpml.print_memory_usage()
print '-'*47

#--------------------------------------------------------------------
# Output
Ez = sc.zeros( (Nx+2, Ny, Nz), 'f' )
'''
psizExf = sc.ones( Cpml.size_z, 'f' )
psizEyf = sc.zeros( Cpml.size_z, 'f' )
psizHxf = sc.zeros( Cpml.size_z, 'f' )
psizHyf = sc.zeros( Cpml.size_z, 'f' )
cuda.memcpy_dtoh( psizExf, Cpml.psizExf )
cuda.memcpy_dtoh( psizEyf, Cpml.psizEyf )
cuda.memcpy_dtoh( psizHxf, Cpml.psizHxf )
cuda.memcpy_dtoh( psizHyf, Cpml.psizHyf )
print (psizExf != 0).sum()
print (psizEyf != 0).sum()
print (psizHxf != 0).sum()
print (psizHyf != 0).sum()
'''

#--------------------------------------------------------------------
# Graphic
from pylab import *
ion()
figure()

Ez[:,:,Nz/2] = 1
imsh = imshow( transpose( Ez[:,:,Nz/2] ),
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
	Cpml.updateE( ( 'fb', 'fb', 'fb' ) )

	Src.updateE( tstep, S.devEz )

	S.updateH()
	Cpml.updateH( ( 'fb', 'fb', 'fb' ) )

	
	if tstep/50*50 == tstep:
		print_elapsed_time( t0, time(), tstep )
		
		cuda.memcpy_dtoh( Ez, S.devEz )
		imsh.set_array( transpose( Ez[:,:,Nz/2] ) )
		png_str = './gpu_png/Ez-%.6d.png' % tstep
		savefig(png_str) 
	

print_elapsed_time( t0, time(), tstep )


S.free_main_in_dev()
S.free_coeff_in_dev()
S.free_coeff()
Cpml.free_psi_in_dev()
Cpml.free_coeff()
