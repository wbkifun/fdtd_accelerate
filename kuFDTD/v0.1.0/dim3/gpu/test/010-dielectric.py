#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.gpu.base import *
from kufdtd.dim3.gpu.matter import Dielectric
from kufdtd.dim3.gpu.source import Source

import pycuda.autoinit


#--------------------------------------------------------------------
Nx, Ny, Nz = 400, 400, 400
dx = 10e-9
tmax = 300

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
print '-'*47
print 'N(%d, %d, %d)' % (S.Nx, S.Ny, S.Nz)
print 'dx = %g' % S.dx
print 'dt = %g' % S.dt
print ''
S.print_kernel_parameters()
S.print_memory_usage()
print '-'*47

#--------------------------------------------------------------------
# Output
Ez = sc.zeros( (Nx+2, Ny, Nz), 'f' )

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

	Src.updateE( tstep, S.devEz )

	S.updateH()

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
