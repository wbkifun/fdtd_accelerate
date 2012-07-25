#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric
from kufdtd.dim3.cpu.cpml_mpi import CpmlNonKappaMpi

from kufdtd.dim3.mpi import *

#--------------------------------------------------------------------
Nx, Ny, Nz = 242, 200, 32
node_Nx_list = [42, 42, 42, 42, 42, 42]
dx = 10e-9
tmax = 1000
Ncore = 8
Npml = 90
pml_apply_direction = ('fb','fb','')

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
	from pylab import *
	ion()
	figure()

	imsh = imshow( transpose( sc.ones( (Nx-2,Ny), 'f' ) ),
					cmap=cm.jet,
					vmin=-0.2, vmax=0.2,
					origin='lower',
					interpolation='bilinear')
	colorbar()

	outputEz = sc.zeros( (Nx-2,Ny), 'f' )


#--------------------------------------------------------------------
from time import *
t0 = time()
for tstep in xrange( 1, tmax+1 ):
	if myrank is not server:
		S.updateE()
		Cpml.updateE()
		mpi_exchange( S.Ey, S.Ez, myrank, target_list, mpi_func_list )
		mpi_exchange_pbc( S.Ey, S.Ez, myrank )

		if myrank == 3:
			S.Ez[30, Ny/2, :] = sc.sin(0.1*tstep)

		S.updateH()
		Cpml.updateH()

	if tstep/30*30 == tstep:
		if myrank is server:
			print_elapsed_time( t0, time(), tstep )
		
			outputEz[0:40,:] = mpi.receive( 1 )
			outputEz[40:80,:] = mpi.receive( 2 )
			outputEz[80:120,:] = mpi.receive( 3 )
			outputEz[120:160,:] = mpi.receive( 4 )
			outputEz[160:200,:] = mpi.receive( 5 )
			outputEz[200:None,:] = mpi.receive( 6 )
			
			imsh.set_array( transpose( outputEz[:,:] ) )
			png_str = './png/Ez-%.6d.png' % tstep
			savefig(png_str) 

		else:
			mpi.send( S.Ez[1:-1,:,Nz/2].copy(), server )

mpi.finalize()
