#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Drude
from kufdtd.dim3.cpu.cpml_mpi import CpmlNonKappaMpi
from kufdtd.dim3.output_mpi import OutputMpi
from kufdtd.dim3.tfsf_mpi import TfsfMpi

from kufdtd.dim3.mpi import *
from kufdtd.dim3.structure import *


#--------------------------------------------------------------------
tmax = 10000
Npml = 15
pml_apply_direction = ( 'fb', 'fb', 'fb' )

Ncore = 1

#--------------------------------------------------------------------
dx = 10e-9
space_dim = ( 2400e-9, 2000e-9, 700e-9 )
wavelength = 600e-9

cylinder1 = Cylinder( 'c1', 
		(1400e-9, 600e-9, 250e-9), (1400e-9, 600e-9, 450e-9), 350e-9,
		['dielectric', 1.5**2, 0, 0] )
cylinder2 = Cylinder( 'c2', 
		(1400e-9, 600e-9, 250e-9), (1400e-9, 600e-9, 450e-9), 250e-9,
		#(1400e-9, 1000e-9, 250e-9), (1400e-9, 1000e-9, 450e-9), 500e-9, 
		['drude', 1.0, 9.97094e15, 4.22253e14] )
		#['drude', 1.0, 9.89371231e15, 2.7804167234e14] )	by jongho
cylinder3 = Cylinder( 'c3', 
		(1400e-9, 1400e-9, 250e-9), (1400e-9, 1400e-9, 450e-9), 350e-9,
		['dielectric', 1.5**2, 0, 0] )
cylinder4 = Cylinder( 'c4', 
		(1400e-9, 1400e-9, 250e-9), (1400e-9, 1400e-9, 450e-9), 250e-9,
		['dielectric', 1e30, 0, 0] )
#structure_list = [cylinder2]
structure_list = [cylinder1, cylinder2, cylinder3, cylinder4]

#--------------------------------------------------------------------
Output_ey = OutputMpi( 'Ey', 
		(None, None, 35), (None, None, 35), (1, 1, 1) ) 
		#(None, 100, None), (None, 100, None), (1, 1, 1) ) 
output_list = [Output_ey]

Src = TfsfMpi( (50, 20, 20), (200, 180, 50),
		('fb','fb','fb'), wavelength, ['normal', 'x'], 0 )
src_list = [Src]

#--------------------------------------------------------------------
Nx, Ny, Nz = sc.int32( sc.ceil( sc.array(space_dim,'f')/dx-1e-3 ) )
node_length_x_list = calc_node_length_x_list( space_dim[0] )
node_Nx_list = sc.int32( sc.array(node_length_x_list,'f')/dx )

if myrank is not server:
	node_length_x = node_length_x_list[myrank-1]
	node_Nx = node_Nx_list[myrank-1]
	S = Drude( node_Nx+2, Ny+2, Nz+2, dx, Ncore )
	S.allocate()

	wrapbox_pt1 = [ sc.array(node_length_x_list[:myrank-1],'f').sum(), 0, 0 ]
	wrapbox_pt2 = [ sc.array(node_length_x_list[:myrank],'f').sum(), space_dim[1], space_dim[2] ]
	length = list( sc.array(wrapbox_pt2) - sc.array(wrapbox_pt1) )
	center_pt = list( sc.array(wrapbox_pt1) + sc.array(length)/2 )
	S.set_wrapbox( wrapbox_pt1, wrapbox_pt2, center_pt, length )
	structure_groups, wrapbox_groups = calc_structure_groups( structure_list )
	S.set_coeff( structure_groups, wrapbox_groups )
	#S.allocate_drude_space( structure_groups, wrapbox_groups )

	Cpml = CpmlNonKappaMpi( node_Nx_list, Npml, pml_apply_direction )
	Cpml.set_space( S )
	Cpml.allocate_psi()
	Cpml.allocate_coeff()

	target_list, mpi_func_list = calc_oddeven( myrank )

	for output in output_list: output.set_space( S, node_Nx_list )
	for src in src_list: src.set_space( S, node_Nx_list )

	for i, obj in enumerate( [S, Cpml] ):
		mpi.send( obj.mem_usage, server, tag=i )

else:
	for output in output_list: output.set_space( 0, node_Nx_list )

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
					vmin=-1.5, vmax=1.5,
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

	if tstep/10*10 == tstep:
		if myrank is not server:
			for output in output_list: output.send()
		else:
			print_elapsed_time( t0, time(), tstep )
		
			imsh.set_array( transpose( Output_ey.gather() ) )
			png_str = './png/Ez-%.6d.png' % tstep
			savefig(png_str) 

