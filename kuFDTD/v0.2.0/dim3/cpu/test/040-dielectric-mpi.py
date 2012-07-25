#!/usr/bin/env python

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.matter import Dielectric

from kufdtd.dim3.mpi import *

#--------------------------------------------------------------------
Nmpi = 3
Nx, Ny, Nz = 240, 200, 32
node_Nx_list = [82, 82, 82]
dx = 10e-9
tmax = 500
Ncore = 8

target_list, mpi_func_list = calc_oddeven( Nmpi, myrank )

#--------------------------------------------------------------------
if myrank is server:
	# Graphic
	from pylab import *
	ion()
	figure()

	imsh = imshow( transpose( sc.ones( (Nx,Ny), 'f' ) ),
					cmap=cm.jet,
					vmin=-0.2, vmax=0.2,
					origin='lower',
					interpolation='bilinear')
	colorbar()

	outputEz = sc.zeros( (Nx,Ny), 'f' )

else:
	S = Dielectric( node_Nx_list[myrank-1], Ny, Nz, dx, Ncore )

	S.allocate_main()
	S.allocate_coeff()
	S.set_coeff()

	print '-'*47
	print 'myrank = %d, N(%d, %d, %d)' % (myrank, S.Nx, S.Ny, S.Nz)
	print 'dx = %g, dt = %g' % (S.dx, S.dt)
	print ''
	S.print_memory_usage()
	print '-'*47

#--------------------------------------------------------------------
from time import *
t0 = time()
for tstep in xrange( 1, tmax+1 ):
	if myrank is not server:
		S.updateE()
		mpi_exchange( S.Ey, S.Ez, myrank, target_list, mpi_func_list )
		mpi_exchange_pbc( S.Ey, S.Ez, myrank )

		if myrank == 2:
			S.Ez[30, Ny/2-50, :] = sc.sin(0.1*tstep)

		S.updateH()

	if tstep/50*50 == tstep:
		if myrank is server:
			print_elapsed_time( t0, time(), tstep )
		
			outputEz[0:80,:] = mpi.receive( 1 )
			outputEz[80:160,:] = mpi.receive( 2 )
			outputEz[160:None,:] = mpi.receive( 3 )
			
			imsh.set_array( transpose( outputEz[:,:] ) )
			png_str = './png/Ez-%.6d.png' % tstep
			savefig(png_str) 

		else:
			mpi.send( S.Ez[1:-1,:,Nz/2].copy(), server )
