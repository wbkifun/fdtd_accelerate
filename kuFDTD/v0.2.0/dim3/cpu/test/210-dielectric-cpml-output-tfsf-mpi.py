#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric
from kufdtd.dim3.cpu.cpml_mpi import CpmlNonKappaMpi
from kufdtd.dim3.output_mpi import OutputMpi
from kufdtd.dim3.tfsf_mpi import TfsfMpi

from kufdtd.dim3.mpi import *

#--------------------------------------------------------------------
Nx, Ny, Nz = 242, 200, 64
node_Nx_list = [82, 82, 82]
dx = 10e-9
tmax = 1000
Ncore = 8
Npml = 15
pml_apply_direction = ( 'fb', 'fb', 'fb' )
wavelength = 600e-9

#Cpml = CpmlNonKappa( Npml, ('fb','fb','fb') )

Output_ey = OutputMpi( 'Ey', node_Nx_list, (None, None, Nz/2), (None, None, Nz/2), (1, 1, 1) ) 
output_list = [Output_ey]

Src = TfsfMpi( node_Nx_list, (50, 50, 20), (180, 150, 40), ('fb','fb','fb'), wavelength, ['normal', 'x'], 0 )
src_list = [Src]

#--------------------------------------------------------------------
if myrank is not server:
	S = Dielectric( node_Nx_list[myrank-1], Ny, Nz, dx, Ncore )
	S.allocate_main()
	S.allocate_coeff()
	S.set_coeff()

	Cpml = CpmlNonKappaMpi( node_Nx_list, Npml, pml_apply_direction )
	Cpml.set_space( S )
	Cpml.allocate_psi()
	Cpml.allocate_coeff()

	target_list, mpi_func_list = calc_oddeven( myrank )

	for output in output_list: output.set_space( S )
	for src in src_list: src.set_space( S )

	for i, obj in enumerate( [S, Cpml] ):
		mpi.send( obj.mem_usage, server, tag=i )

else:
	mem_usage_list = [ [], [] ]
	for i in xrange( 2 ):
		for rank in xrange( 1, Nnode+1 ):
			mem_usage_list[i].append( mpi.receive( rank, tag=i ) )

	print '-'*47
	print 'N (%d, %d, %d)' % ( Nx, Ny, Nz )
	print 'node_Nx_list =', node_Nx_list
	print 'dx = %g, dt = %g' % ( dx, dx/(2*light_velocity) )
	print 'tmax = %d' % tmax
	print 'Npml = %d' % Npml
	print 'Nnode = %d' % Nnode
	print ''
	print 'memory usage:'
	print_mem_usage( mem_usage_list )
	print '-'*47

	# Graphic
	data_shape = Output_ey.get_shape( Nx, Ny, Nz )
	print data_shape

	from pylab import *
	ion()
	figure()

	imsh = imshow( transpose( sc.ones( data_shape, 'f' ) ),
					cmap=cm.jet,
					vmin=-0.05, vmax=0.05,
					origin='lower',
					interpolation='bilinear')
	colorbar()


#--------------------------------------------------------------------
from time import *
t0 = time()
for tstep in xrange( 1, tmax+1 ):
	if myrank is not server:
		S.updateE()
		Cpml.updateE()
		Src.updateE( tstep )
		mpi_exchange( S.Ey, S.Ez, myrank, target_list, mpi_func_list )
		mpi_exchange_pbc( S.Ey, S.Ez, myrank )

		S.updateH()
		Cpml.updateH()
		Src.updateH()

	if tstep/50*50 == tstep:
		if myrank is not server:
			for output in output_list: output.send()
		else:
			print_elapsed_time( t0, time(), tstep )
		
			imsh.set_array( transpose( Output_ey.gather() ) )
			png_str = './png/Ez-%.6d.png' % tstep
			savefig(png_str) 

