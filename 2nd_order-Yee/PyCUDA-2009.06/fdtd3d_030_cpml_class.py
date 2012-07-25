#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit

import scipy as sc
import sys

light_velocity = 2.99792458e8	# m s- 
ep0 = 8.85418781762038920e-12	# F m-1 (permittivity at vacuum)
mu0 = 1.25663706143591730e-6	# N A-2 (permeability at vacuum)
imp0 = sc.sqrt( mu0/ep0 )		# (impedance at vacuum)
pi = 3.14159265358979323846

def print_elapsed_time( t1 ):
	elapse_time = localtime(t1-t0-60*60*9)
	str_time = strftime('[%j]%H:%M:%S', elapse_time)
	print '%s    tstep = %d' % (str_time, tstep)


class FdtdSpace:
	def __init__( s, Nx, Ny, Nz, dx ):
		s.Nx = Nx
		s.Ny = Ny
		s.Nz = Nz
		s.dx = dx

		courant = 0.5		# Courant factor
		s.dt = courant*dx/light_velocity



class Dielectric():
	def allocate_coeff( s ):
		shape = (s.Nx+1, s.Ny, s.Nz)
		s.CEx = sc.zeros( shape, 'f' )
		s.CEy = sc.zeros( shape, 'f' )
		s.CEz = sc.zeros( shape, 'f' )


	def free_coeff( s ):
		del s.CEx
		del s.CEy
		del s.CEz 


	def set_coeff( s ):
		s.CEx[1:,1:-1,1:-1] = 0.5
		s.CEy[1:-1,1:,1:-1] = 0.5
		s.CEz[1:-1,1:-1,1:] = 0.5

	

class GpuSpace( FdtdSpace ):
	def __init__( s, Nx, Ny, Nz, dx ):
		FdtdSpace.__init__( s, Nx, Ny, Nz, dx )

		s.bytes_f = sc.zeros(1,'f').nbytes

		s.kNx = sc.int32(s.Nx)
		s.kNy = sc.int32(s.Ny)
		s.kNz = sc.int32(s.Nz)


	def select_tpb( s ):
		pass


	def calc_bpg( s, N, tpb ):
		if ( N%tpb == 0 ):
			bpg = N/tpb
		else:
			bpg = N/tpb + 1

		return bpg


	def get_kernel_initmem( s ):
		mod = cuda.SourceModule( file('./gpu_core/initmem.cu','r').read() )
		return mod.get_function("initmem")



class MatterGpu( GpuSpace ):
	def __init__( s, Nx, Ny, Nz, dx ):
		GpuSpace.__init__( s, Nx, Ny, Nz, dx )

		s.size = s.Nx*s.Ny*s.Nz
		s.size1 = ( s.Nx+1 )*s.Ny*s.Nz
		s.size2 = ( s.Nx+2 )*s.Ny*s.Nz

		s.bytes1 = s.size1*s.bytes_f
		s.bytes2 = s.size2*s.bytes_f

		s.verify_16xNz()


	def verify_16xNz( s ):
		R = s.Nz%16
		if ( R == 0 ):
			print '-'*47
			print 'Nz is a multiple of 16.'
		else:
			print '-'*47
			print 'Error: Nz is not a multiple of 16.'
			print 'Recommend Nz: %d or %d' % (s.Nz-R, s.Nz-R+16)
			sys.exit(0)


	def allocate_main_in_dev( s ):
		s.devEx = cuda.mem_alloc( s.bytes2 )
		s.devEy = cuda.mem_alloc( s.bytes2 )
		s.devEz = cuda.mem_alloc( s.bytes2 )
		s.devHx = cuda.mem_alloc( s.bytes2 )
		s.devHy = cuda.mem_alloc( s.bytes2 )
		s.devHz = cuda.mem_alloc( s.bytes2 )


	def free_main_in_dev( s ):
		s.devEx.free()
		s.devEy.free()
		s.devEz.free()
		s.devHx.free()
		s.devHy.free()
		s.devHz.free()


	def initmem_main_in_dev( s ):
		initmem = s.get_kernel_initmem()

		tpb = 512
		bpg = s.calc_bpg( s.size2, tpb )
		N = sc.int32( s.size2 )
		Db = (tpb,1,1)
		Dg = (bpg,1)

		initmem( N, s.devEx, block=Db, grid=Dg )
		initmem( N, s.devEy, block=Db, grid=Dg )
		initmem( N, s.devEz, block=Db, grid=Dg )
                              
		initmem( N, s.devHx, block=Db, grid=Dg )
		initmem( N, s.devHy, block=Db, grid=Dg )
		initmem( N, s.devHz, block=Db, grid=Dg )
		                      

	def allocate_coeff_in_dev( s ):
		s.devCEx = cuda.mem_alloc( s.bytes1 )
		s.devCEy = cuda.mem_alloc( s.bytes1 )
		s.devCEz = cuda.mem_alloc( s.bytes1 )


	def free_coeff_in_dev( s ):
		s.devCEx.free()
		s.devCEy.free()
		s.devCEz.free()



class DielectricGpu( Dielectric, MatterGpu ):
	def __init__( s, Nx, Ny, Nz, dx ):
		MatterGpu.__init__( s, Nx, Ny, Nz, dx )

		s.set_kernel_parameters()


	def set_kernel_parameters( s ):
		s.tpb_main = 512
		s.bpg_main = s.calc_bpg( s.size, s.tpb_main )
		s.ns_main = ( 2*(s.tpb_main+1)+s.tpb_main )*s.bytes_f



	def print_main_kernel_parameters( s ):
		print 'main: tpb=%d, bpg=%d, ns=%d' % (s.tpb_main, s.bpg_main, s.ns_main)


	def memcpy_htod_coeff( s ):
		cuda.memcpy_htod( s.devCEx, s.CEx )
		cuda.memcpy_htod( s.devCEy, s.CEy )
		cuda.memcpy_htod( s.devCEz, s.CEz )


	def prepare_kernels( s ):
		mod = cuda.SourceModule( file('./gpu_core/dielectric.cu','r').read() )
		s.update_e = mod.get_function("update_e")
		s.update_h = mod.get_function("update_h")

		Db = ( s.tpb_main, 1, 1 )
		s.update_e.prepare( "iiPPPPPPPPP", block=Db, shared=s.ns_main )
		s.update_h.prepare( "iiPPPPPP", block=Db, shared=s.ns_main )


	def updateE( s ):
		s.update_e.prepared_call( \
				( s.bpg_main, 1 ), \
				s.kNy, s.kNz, \
				s.devEx, s.devEy, s.devEz, \
				s.devHx, s.devHy, s.devHz, 
				s.devCEx, s.devCEy, s.devCEz )
	

	def updateH( s ):
		s.update_h.prepared_call( \
				( s.bpg_main, 1 ), \
				s.kNy, s.kNz, \
				s.devEx, s.devEy, s.devEz, \
				s.devHx, s.devHy, s.devHz )



class CpmlNonKapaGpu( GpuSpace ):
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
		s.tpb_x = 512
		s.bpg_x = s.calc_bpg( s.Npml*s.Ny*s.Nz, s.tpb_x )

		s.tpb_y = 512
		s.bpg_y = s.calc_bpg( s.Nx*s.Npml*s.Nz, s.tpb_y )

		s.tpb_z = 512
		s.bpg_z = s.calc_bpg( s.Nx*s.Ny*(s.Npml+1), s.tpb_z )
		s.ns_z = ( 2*(s.tpb_z+1) )*s.bytes_f


	def print_cpml_kernel_parameters( s ):
		print 'cpml x: tpb=%d, bpg=%d' % (s.tpb_x, s.bpg_x)
		print 'cpml y: tpb=%d, bpg=%d' % (s.tpb_y, s.bpg_y)
		print 'cpml z: tpb=%d, bpg=%d, ns=%d' % (s.tpb_z, s.bpg_z, s.ns_z)


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
		sigma_max = (m+1.)/(15*pi*s.Npml*dx)
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
		s.mod = cuda.SourceModule( file('./gpu_core/cpml_non_kapa.cu','r').read().replace('NPMLp2',str(2*(Npml+1))).replace('NPMLp',str(Npml+1)).replace('NPML',str(Npml)  ) )


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
		s.update_cpml_x_E = s.mod.get_function("update_cpml_x_E")
		s.update_cpml_x_H = s.mod.get_function("update_cpml_x_H")
		s.update_cpml_y_E = s.mod.get_function("update_cpml_y_E")
		s.update_cpml_y_H = s.mod.get_function("update_cpml_y_H")
		s.update_cpml_z_E = s.mod.get_function("update_cpml_z_E")
		s.update_cpml_z_H = s.mod.get_function("update_cpml_z_H")


		Db = ( s.tpb_x, 1, 1 )
		s.update_cpml_x_E.prepare( "iiiPPPPPPPPPPPi", block=Db )
		s.update_cpml_x_H.prepare( "iiiPPPPPPPPi", block=Db )

		Db = ( s.tpb_y, 1, 1 )
		s.update_cpml_y_E.prepare( "iiPPPPPPPPPPPi", block=Db )
		s.update_cpml_y_H.prepare( "iiPPPPPPPPi", block=Db )

		Db = ( s.tpb_z, 1, 1 )
		s.update_cpml_z_E.prepare( "iiPPPPPPPPPPPi", block=Db, shared=s.ns_z )
		s.update_cpml_z_H.prepare( "iiPPPPPPPPi", block=Db, shared=s.ns_z )


	def updateE( s, direction ):
		if 'f' in direction[0]:
			s.update_cpml_x_E.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psixEyf, s.psixEzf, \
					0 )

		if 'b' in direction[0]:
			s.update_cpml_x_E.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psixEyb, s.psixEzb, \
					1 )

		if 'f' in direction[1]:
			s.update_cpml_y_E.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psiyEzf, s.psiyExf, \
					0 )

		if 'b' in direction[1]:
			s.update_cpml_y_E.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psiyEzb, s.psiyExb, \
					1 )

		if 'f' in direction[2]:
			s.update_cpml_z_E.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psizExf, s.psizEyf, \
					0 )

		if 'b' in direction[2]:
			s.update_cpml_z_E.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.devCEx, s.devCEy, s.devCEz, \
					s.psizExb, s.psizEyb, \
					1 )


	def updateH( s, direction ):
		if 'f' in direction[0]:
			s.update_cpml_x_H.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psixHyf, s.psixHzf, \
					0 )

		if 'b' in direction[0]:
			s.update_cpml_x_H.prepared_call( \
					( s.bpg_x, 1 ), \
					s.kNx, s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psixHyb, s.psixHzb, \
					1 )

		if 'f' in direction[1]:
			s.update_cpml_y_H.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psiyHzf, s.psiyHxf, \
					0 )

		if 'b' in direction[1]:
			s.update_cpml_y_H.prepared_call( \
					( s.bpg_y, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psiyHzb, s.psiyHxb, \
					1 )

		if 'f' in direction[2]:
			s.update_cpml_z_H.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psizHxf, s.psizHyf, \
					0 )

		if 'b' in direction[2]:
			s.update_cpml_z_H.prepared_call( \
					( s.bpg_z, 1 ), \
					s.kNy, s.kNz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, \
					s.psizHxb, s.psizHyb, \
					1 )


class Source( GpuSpace ):
	def __init__( s, main_space ):
		MS = main_space
		GpuSpace.__init__( s, MS.Nx, MS.Ny, MS.Nz, MS.dx )

		s.set_kernel_parameters()


	def set_kernel_parameters( s ):
		s.tpb = s.Nz
		s.bpg = 1


	def prepare_kernels( s ):
		mod = cuda.SourceModule( file('./gpu_core/source.cu','r').read() )
		s.update_src = mod.get_function("update_src")

		Db = ( s.tpb, 1, 1 )
		s.update_src.prepare( "iiiiP", block=Db )


	def updateE( s, tstep, F ):
		s.update_src.prepared_call( (s.bpg,1), s.kNx, s.kNy, s.kNz, sc.int32(tstep), F )




#--------------------------------------------------------------------
# test
#--------------------------------------------------------------------
if __name__ == '__main__':
	Nx, Ny, Nz = 250, 250, 320
	dx = 10e-9

	#--------------------------------------------------------------------
	S = DielectricGpu( Nx, Ny, Nz, dx )

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

	Cpml = CpmlNonKapaGpu( Npml, S )
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
	S.print_main_kernel_parameters()
	Cpml.print_cpml_kernel_parameters()
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
	'''
	#--------------------------------------------------------------------
	from time import *
	t0 = time()
	for tstep in xrange( 1, 100001 ):
		S.updateE()
		Cpml.updateE( ( 'fb', 'fb', 'fb' ) )

		Src.updateE( tstep, S.devEz )

		S.updateH()
		Cpml.updateH( ( 'fb', 'fb', 'fb' ) )
	
		'''
		if tstep/100*100 == tstep:
			print_elapsed_time( time() )
			
			cuda.memcpy_dtoh( Ez, S.devEz )
			imsh.set_array( transpose( Ez[:,:,Nz/2] ) )
			png_str = './gpu_png/Ez-%.6d.png' % tstep
			savefig(png_str) 
		'''

	print_elapsed_time( time() )


	S.free_main_in_dev()
	S.free_coeff_in_dev()
	S.free_coeff()
	Cpml.free_psi_in_dev()
	Cpml.free_coeff()
