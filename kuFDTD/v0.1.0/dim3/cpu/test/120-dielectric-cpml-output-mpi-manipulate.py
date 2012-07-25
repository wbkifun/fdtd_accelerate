#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric
from kufdtd.dim3.cpu.cpml import CpmlNonKappa
from kufdtd.dim3.mpi import *
from kufdtd.dim3.output import Output
from kufdtd.dim3.output_mpi import OutputMpiServer, OutputMpiNode
from kufdtd.dim3.output_manipulate import ViewGraphic2d


#--------------------------------------------------------------------
Nx, Ny, Nz = 240, 200, 32
node_Nx_list = [82, 82, 82]
dx = 10e-9
tmax = 1000
Ncore = 8

Npml = 15

# output data
pt1 = ( None, None, Nz/2 )
pt2 = ( None, None, Nz/2 )
spartial_step = ( 2, 2, 2 )

#--------------------------------------------------------------------

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

	print '-'*47
	print 'myrank = %d, N(%d, %d, %d)' % (myrank, S.Nx, S.Ny, S.Nz)
	print 'dx = %g, dt = %g' % (S.dx, S.dt)
	print 'Npml = %g' % (Cpml.Npml)
	print ''
	S.print_memory_usage()
	Cpml.print_memory_usage()
	print '-'*47

	Output_ez = OutputMpiNode( S, 'Ez', myrank, node_Nx_list, pt1, pt2, spartial_step ) 

else:
	Output_ez = OutputMpiServer( node_Nx_list, pt1, pt2, spartial_step )


if myrank is not server: 
	Output_ez.send()
else: 
	data_shape = Output_ez.gather().shape
	print data_shape
	g_extra_args = ['colorbar']
	graphic = ViewGraphic2d( data_shape, g_extra_args )


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
			Output_ez.send()
		else:
			print_elapsed_time( t0, time(), tstep )
		
			graphic.draw( Output_ez.gather() )
			png_path = './png/Ez-%.6d.png' % tstep
			graphic.save( png_path ) 
