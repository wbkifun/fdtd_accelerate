#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric
from kufdtd.dim3.cpu.cpml import CpmlNonKappa
from kufdtd.dim3.output_mpi import OutputMpi

from kufdtd.dim3.mpi import *

#--------------------------------------------------------------------
Nx, Ny, Nz = 242, 200, 32
node_Nx_list = [82, 82, 82]
dx = 10e-9
tmax = 1000
Ncore = 8

Npml = 15

# output data
pt1 = (None, None, Nz/2)
pt2 = ( None, None, Nz/2 )
spartial_step = ( 1, 1, 1 )

#--------------------------------------------------------------------
Output_ez = OutputMpi( 'Ez', node_Nx_list, pt1, pt2, spartial_step ) 
Output_ex = OutputMpi( 'Ex', node_Nx_list, pt1, pt2, spartial_step ) 

Output_ez_yz = OutputMpi( 'Ez', node_Nx_list, (Nx/2, None, None), (Nx/2, None, None), spartial_step ) 
Output_ez_yz_2 = OutputMpi( 'Ez', node_Nx_list, (Nx/2, 10, None), (Nx/2, 100, None), spartial_step ) 
#Output_ez_yz_2 = OutputMpi( 'Ez', node_Nx_list, (Nx/2, 10, 20), (Nx/2, 100, 20), spartial_step ) 
output_list = [Output_ez, Output_ex, Output_ez_yz, Output_ez_yz_2]

#--------------------------------------------------------------------
if myrank is not server:
	S = Dielectric( node_Nx_list[myrank-1], Ny, Nz, dx, Ncore )

	S.allocate_main()
	S.allocate_coeff()
	S.set_coeff()

	Cpml = CpmlNonKappa( Npml, S )
	Cpml.allocate_psi()
	Cpml.allocate_coeff()

	if myrank == 1:
		pml_direction = ( 'f', 'fb', 'fb' )
	elif myrank == Nnode:
		pml_direction = ( 'b', 'fb', 'fb' )
	else:
		pml_direction = ( '', 'fb', 'fb' )

	target_list, mpi_func_list = calc_oddeven( myrank )
	for output in output_list: output.set_space( S )

	print '-'*47
	print 'myrank = %d, N(%d, %d, %d)' % (myrank, S.Nx, S.Ny, S.Nz)
	print 'dx = %g, dt = %g' % (S.dx, S.dt)
	print 'Npml = %g' % (Cpml.Npml)
	print ''
	S.print_memory_usage()
	Cpml.print_memory_usage()
	print '-'*47



if myrank is server: 
	data_shape = Output_ez.get_shape( Nx, Ny, Nz )
	print data_shape

	# Graphic
	from pylab import *
	ion()
	figure()

	imsh = imshow( transpose( sc.ones( data_shape, 'f' ) ),
					cmap=cm.jet,
					vmin=-0.05, vmax=0.05,
					origin='lower',
					interpolation='bilinear')
	colorbar()
'''
from pylab import *
ion()
'''


#--------------------------------------------------------------------
from time import *
t0 = time()
for tstep in xrange( 1, tmax+1 ):
	if myrank is not server:
		S.updateE()
		Cpml.updateE( pml_direction )
		mpi_exchange( S.Ey, S.Ez, myrank, target_list, mpi_func_list )
		mpi_exchange_pbc( S.Ey, S.Ez, myrank )

		if myrank == 2:
			S.Ez[30, Ny/2-50, :] += sc.sin(0.1*tstep)

		S.updateH()
		Cpml.updateH( pml_direction )


	if tstep/50*50 == tstep:
		if myrank is not server:
			for output in output_list: output.send()
		else:
			print_elapsed_time( t0, time(), tstep )
		
			imsh.set_array( transpose( Output_ez.gather() ) )
			Output_ex.gather()
			Output_ez_yz.gather()
			Output_ez_yz_2.gather()
			#plot( Output_ez_yz_2.gather() )
			png_str = './png/Ez-%.6d.png' % tstep
			savefig(png_str) 

