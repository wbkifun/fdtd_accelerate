'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.gpu.base import *


class CpmlNonKapa( GpuSpace ):
	def __init__( s, Npml, main_space ):
		MS = main_space
		GpuSpace.__init__( s, MS.Nx, MS.Ny, MS.Nz, MS.dx )
		s.devEx = MS.devEx
		s.devEy = MS.devEy
		s.devEz = MS.devEz
		s.devHx = MS.devHx
		s.devHy = MS.devHy
		s.devHz = MS.devHz
		s.devCEx = MS.devCEx
		s.devCEy = MS.devCEy
		s.devCEz = MS.devCEz
		
		s.Npml = Npml

		s.set_kernel_parameters()
		s.size_x = s.tpb_x*s.bpg_x
		s.size_y = s.tpb_y*s.bpg_y
		s.size_z = s.tpb_z*s.bpg_z

		s.bytes_x = s.size_x*s.bytes_f
		s.bytes_y = s.size_y*s.bytes_f
		s.bytes_z = s.size_z*s.bytes_f


	def set_kernel_parameters( s ):
		Ntot_x = s.Npml*s.Ny*s.Nz
		s.tpb_x, s.occupancy_x = s.select_tpb( Ntot_x, s.Ny*s.Nz )
		s.bpg_x = s.calc_bpg( Ntot_x, s.tpb_x )

		Ntot_y = s.Nx*s.Npml*s.Nz
		s.tpb_y, s.occupancy_y = s.select_tpb( Ntot_y, s.Npml*s.Nz )
		s.bpg_y = s.calc_bpg( Ntot_y, s.tpb_y )

		Ntot_z = s.Nx*s.Ny*(s.Npml+1)
		s.tpb_z, s.occupancy_z = s.select_tpb( Ntot_z, s.Ny*(s.Npml+1) )
		s.bpg_z = s.calc_bpg( Ntot_z, s.tpb_z )
		s.ns_z = ( 2*(s.tpb_z+1) )*s.bytes_f


	def print_kernel_parameters( s ):
		print 'cpml x: tpb=%d, bpg=%d, occupancy=%1.2f' % (s.tpb_x, s.bpg_x, s.occupancy_x)
		print 'cpml y: tpb=%d, bpg=%d, occupancy=%1.2f' % (s.tpb_y, s.bpg_y, s.occupancy_y)
		print 'cpml z: tpb=%d, bpg=%d, occupancy=%1.2f, ns=%d' % (s.tpb_z, s.bpg_z, s.occupancy_z, s.ns_z)


	def print_memory_usage( s ):
		mbytes = 1024**2
		psix = 8*s.bytes_x/mbytes
		psiy = 8*s.bytes_y/mbytes
		psiz = 8*s.bytes_z/mbytes

		print 'memory usage: %d Mbytes (psi: x=%d, y=%d, z=%d)' % ( psix+psiy+psiz, psix, psiy, psiz ) 
		
	def allocate_psi_in_dev( s ):
		s.psixEyf = cuda.mem_alloc( s.bytes_x )
		s.psixEyb = cuda.mem_alloc( s.bytes_x )
		s.psixEzf = cuda.mem_alloc( s.bytes_x )
		s.psixEzb = cuda.mem_alloc( s.bytes_x )
		s.psixHyf = cuda.mem_alloc( s.bytes_x )
		s.psixHyb = cuda.mem_alloc( s.bytes_x )
		s.psixHzf = cuda.mem_alloc( s.bytes_x )
		s.psixHzb = cuda.mem_alloc( s.bytes_x )

		s.psiyEzf = cuda.mem_alloc( s.bytes_y )
		s.psiyEzb = cuda.mem_alloc( s.bytes_y )
		s.psiyExf = cuda.mem_alloc( s.bytes_y )
		s.psiyExb = cuda.mem_alloc( s.bytes_y )
		s.psiyHzf = cuda.mem_alloc( s.bytes_y )
		s.psiyHzb = cuda.mem_alloc( s.bytes_y )
		s.psiyHxf = cuda.mem_alloc( s.bytes_y )
		s.psiyHxb = cuda.mem_alloc( s.bytes_y )

		s.psizExf = cuda.mem_alloc( s.bytes_z )
		s.psizExb = cuda.mem_alloc( s.bytes_z )
		s.psizEyf = cuda.mem_alloc( s.bytes_z )
		s.psizEyb = cuda.mem_alloc( s.bytes_z )
		s.psizHxf = cuda.mem_alloc( s.bytes_z )
		s.psizHxb = cuda.mem_alloc( s.bytes_z )
		s.psizHyf = cuda.mem_alloc( s.bytes_z )
		s.psizHyb = cuda.mem_alloc( s.bytes_z )


	def free_psi_in_dev( s ):
		s.psixEyf.free()
		s.psixEyb.free()
		s.psixEzf.free()
		s.psixEzb.free()
		s.psixHyf.free()
		s.psixHyb.free()
		s.psixHzf.free()
		s.psixHzb.free()

		s.psiyEzf.free()
		s.psiyEzb.free()
		s.psiyExf.free()
		s.psiyExb.free()
		s.psiyHzf.free()
		s.psiyHzb.free()
		s.psiyHxf.free()
		s.psiyHxb.free()

		s.psizExf.free()
		s.psizExb.free()
		s.psizEyf.free()
		s.psizEyb.free()
		s.psizHxf.free()
		s.psizHxb.free()
		s.psizHyf.free()
		s.psizHyb.free()


	def initmem_psi_in_dev( s ):
		initmem = s.get_kernel_initmem()

		N = sc.int32( s.size_x )
		Db = ( s.tpb_x, 1, 1 )
		Dg = ( s.bpg_x, 1 )
		initmem( N, s.psixEyf, block=Db, grid=Dg )
		initmem( N, s.psixEyb, block=Db, grid=Dg )
		initmem( N, s.psixEzf, block=Db, grid=Dg )
		initmem( N, s.psixEzb, block=Db, grid=Dg )
		initmem( N, s.psixHyf, block=Db, grid=Dg )
		initmem( N, s.psixHyb, block=Db, grid=Dg )
		initmem( N, s.psixHzf, block=Db, grid=Dg )
		initmem( N, s.psixHzb, block=Db, grid=Dg )

		N = sc.int32( s.size_y )
		Db = ( s.tpb_y, 1, 1 )
		Dg = ( s.bpg_y, 1 )
		initmem( N, s.psiyEzf, block=Db, grid=Dg )
		initmem( N, s.psiyEzb, block=Db, grid=Dg )
		initmem( N, s.psiyExf, block=Db, grid=Dg )
		initmem( N, s.psiyExb, block=Db, grid=Dg )
		initmem( N, s.psiyHzf, block=Db, grid=Dg )
		initmem( N, s.psiyHzb, block=Db, grid=Dg )
		initmem( N, s.psiyHxf, block=Db, grid=Dg )
		initmem( N, s.psiyHxb, block=Db, grid=Dg )

		N = sc.int32( s.size_z )
		Db = ( s.tpb_z, 1, 1 )
		Dg = ( s.bpg_z, 1 )
		initmem( N, s.psizExf, block=Db, grid=Dg )
		initmem( N, s.psizExb, block=Db, grid=Dg )
		initmem( N, s.psizEyf, block=Db, grid=Dg )
		initmem( N, s.psizEyb, block=Db, grid=Dg )
		initmem( N, s.psizHxf, block=Db, grid=Dg )
		initmem( N, s.psizHxb, block=Db, grid=Dg )
		initmem( N, s.psizHyf, block=Db, grid=Dg )
		initmem( N, s.psizHyb, block=Db, grid=Dg )


	def allocate_coeff( s ):
		s.bE = sc.zeros( 2*(s.Npml+1), 'f')
		s.bH = sc.zeros( 2*(s.Npml+1), 'f')
		s.aE = sc.zeros( 2*(s.Npml+1), 'f')
		s.aH = sc.zeros( 2*(s.Npml+1), 'f')


	def free_coeff( s ):
		del s.bE
		del s.bH
		del s.aE
		del s.aH


	def set_coeff( s ):
		m = 4	# grade_order
		sigma_max = (m+1.)/(15*pi*s.Npml*s.dx)
		alpha = 0.05

		sigmaE = sc.zeros( 2*(s.Npml+1), 'f')
		sigmaH = sc.zeros( 2*(s.Npml+1), 'f')
		for i in xrange(s.Npml):
			sigmaE[i] = pow( (s.Npml-0.5-i)/s.Npml, m )*sigma_max
			sigmaE[i+s.Npml+1] = pow( (0.5+i)/s.Npml, m )*sigma_max
			sigmaH[i+1] = pow( float(s.Npml-i)/s.Npml, m )*sigma_max
			sigmaH[i+s.Npml+2] = pow( (1.+i)/s.Npml, m )*sigma_max

		s.bE[:] = sc.exp( -(sigmaE[:] + alpha)*s.dt/ep0 );
		s.bH[:] = sc.exp( -(sigmaH[:] + alpha)*s.dt/ep0 );
		s.aE[:] = sigmaE[:]/(sigmaE[:]+alpha)*(s.bE[:]-1);
		s.aH[:] = sigmaH[:]/(sigmaH[:]+alpha)*(s.bH[:]-1);
		
		del sigmaE, sigmaH


	def get_module( s ):
		fpath = '%s/core/cpml_non_kapa.cu' % base_dir
		s.mod = cuda.SourceModule( file( fpath,'r' ).read().replace('NPMLp2',str(2*(s.Npml+1))).replace('NPMLp',str(s.Npml+1)).replace('NPML',str(s.Npml)  ) )


	def memcpy_htod_coeff( s ):
		rcmbE = s.mod.get_global("rcmbE")
		rcmbH = s.mod.get_global("rcmbH")
		rcmaE = s.mod.get_global("rcmaE")
		rcmaH = s.mod.get_global("rcmaH")

		cuda.memcpy_htod( rcmbE[0], s.bE )
		cuda.memcpy_htod( rcmbH[0], s.bH )
		cuda.memcpy_htod( rcmaE[0], s.aE )
		cuda.memcpy_htod( rcmaH[0], s.aH )

	
	def prepare_kernels( s ):
		s.update_x_e = s.mod.get_function("update_x_e")
		s.update_x_h = s.mod.get_function("update_x_h")
		s.update_y_e = s.mod.get_function("update_y_e")
		s.update_y_h = s.mod.get_function("update_y_h")
		s.update_z_e = s.mod.get_function("update_z_e")
		s.update_z_h = s.mod.get_function("update_z_h")


		Db = ( s.tpb_x, 1, 1 )
		s.update_x_e.prepare( "iiiPPPPPPPPPPPi", block=Db )
		s.update_x_h.prepare( "iiiPPPPPPPPi", block=Db )

		Db = ( s.tpb_y, 1, 1 )
		s.update_y_e.prepare( "iiPPPPPPPPPPPi", block=Db )
		s.update_y_h.prepare( "iiPPPPPPPPi", block=Db )

		Db = ( s.tpb_z, 1, 1 )
		s.update_z_e.prepare( "iiPPPPPPPPPPPi", block=Db, shared=s.ns_z )
		s.update_z_h.prepare( "iiPPPPPPPPi", block=Db, shared=s.ns_z )


	def updateE( s, direction ):
		if 'f' in direction[0]:
			s.update_x_e.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psixEyf, s.psixEzf, \
					0 )

		if 'b' in direction[0]:
			s.update_x_e.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psixEyb, s.psixEzb, \
					1 )

		if 'f' in direction[1]:
			s.update_y_e.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psiyEzf, s.psiyExf, \
					0 )

		if 'b' in direction[1]:
			s.update_y_e.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psiyEzb, s.psiyExb, \
					1 )

		if 'f' in direction[2]:
			s.update_z_e.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psizExf, s.psizEyf, \
					0 )

		if 'b' in direction[2]:
			s.update_z_e.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psizExb, s.psizEyb, \
					1 )


	def updateH( s, direction ):
		if 'f' in direction[0]:
			s.update_x_h.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psixHyf, s.psixHzf, \
					0 )

		if 'b' in direction[0]:
			s.update_x_h.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psixHyb, s.psixHzb, \
					1 )

		if 'f' in direction[1]:
			s.update_y_h.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psiyHzf, s.psiyHxf, \
					0 )

		if 'b' in direction[1]:
			s.update_y_h.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psiyHzb, s.psiyHxb, \
					1 )

		if 'f' in direction[2]:
			s.update_z_h.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psizHxf, s.psizHyf, \
					0 )

		if 'b' in direction[2]:
			s.update_z_h.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psizHxb, s.psizHyb, \
					1 )
