'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 7. 18
 last update  :

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.core import cpml_non_kappa


class CpmlNonKappa( CpuSpace ):
	def __init__( s, Npml, apply_direction ):
		s.Npml = s.Npmlx = Npml
		s.apply_direction = list( apply_direction )

		if len( apply_direction[2] ) > 0:
			s.verify_4xNz( Npml+1 )

		s.forward = 0
		s.backward = 1


	def set_space( s, Space ):
		CpuSpace.__init__( s, Space.Nx, Space.Ny, Space.Nz, Space.dx, Space.Ncore )
		s.Ex, s.Ey, s.Ez = Space.Ex, Space.Ey, Space.Ez
		s.Hx, s.Hy, s.Hz = Space.Hx, Space.Hy, Space.Hz
		s.CEx, s.CEy, s.CEz = Space.CEx, Space.CEy, Space.CEz


	def allocate_psi( s ):
		shape_y = (s.Nx-1, s.Npml, s.Nz)
		shape_z = (s.Nx-1, s.Ny-1, s.Npml+1)

		if 'f' in s.apply_direction[1]:
			s.psiyEzf = sc.zeros( shape_y, 'f' )
			s.psiyExf = sc.zeros( shape_y, 'f' )
			s.psiyHzf = sc.zeros( shape_y, 'f' )
			s.psiyHxf = sc.zeros( shape_y, 'f' )
			s.mem_usage += 4*s.psiyEzf.nbytes

		if 'b' in s.apply_direction[1]:
			s.psiyEzb = sc.zeros( shape_y, 'f' )
			s.psiyExb = sc.zeros( shape_y, 'f' )
			s.psiyHzb = sc.zeros( shape_y, 'f' )
			s.psiyHxb = sc.zeros( shape_y, 'f' )
			s.mem_usage += 4*s.psiyEzb.nbytes

		if 'f' in s.apply_direction[2]:
			s.psizExf = sc.zeros( shape_z, 'f' )
			s.psizEyf = sc.zeros( shape_z, 'f' )
			s.psizHxf = sc.zeros( shape_z, 'f' )
			s.psizHyf = sc.zeros( shape_z, 'f' )
			s.mem_usage += 4*s.psizExf.nbytes

		if 'b' in s.apply_direction[2]:
			s.psizExb = sc.zeros( shape_z, 'f' )
			s.psizEyb = sc.zeros( shape_z, 'f' )
			s.psizHxb = sc.zeros( shape_z, 'f' )
			s.psizHyb = sc.zeros( shape_z, 'f' )
			s.mem_usage += 4*s.psizExb.nbytes

		s.allocate_psix( s.Npmlx )


	def allocate_psix( s, Npmlx ):
		shape_x = (Npmlx, s.Ny-1, s.Nz)

		if 'f' in s.apply_direction[0]:
			s.psixEyf = sc.zeros( shape_x, 'f' )
			s.psixEzf = sc.zeros( shape_x, 'f' )
			s.psixHyf = sc.zeros( shape_x, 'f' )
			s.psixHzf = sc.zeros( shape_x, 'f' )
			s.mem_usage += 4*s.psixEyf.nbytes

		if 'b' in s.apply_direction[0]:
			s.psixEyb = sc.zeros( shape_x, 'f' )
			s.psixEzb = sc.zeros( shape_x, 'f' )
			s.psixHyb = sc.zeros( shape_x, 'f' )
			s.psixHzb = sc.zeros( shape_x, 'f' )
			s.mem_usage += 4*s.psixEyb.nbytes


	def allocate_coeff( s ):
		s.bE = sc.zeros( 2*(s.Npml+1), 'f')
		s.aE = sc.zeros( 2*(s.Npml+1), 'f')
		s.bH = sc.zeros( 2*(s.Npml+1), 'f')
		s.aH = sc.zeros( 2*(s.Npml+1), 'f')

		s.calc_coeff( s.bE, s.aE, s.bH, s.aH )
		
		s.allocate_coeffx( s.Npmlx )
	

	def allocate_coeffx( s, Npmlx ):
		if Npmlx == s.Npml:
			s.bEx = s.bE
			s.aEx = s.aE
			s.bHx = s.bH
			s.aHx = s.aH


	def calc_coeff( s, bE, aE, bH, aH ):
		m = 4	# grade_order
		sigma_max = (m+1.)/(15*pi*s.Npml*s.dx)
		alpha = 0.05

		sigmaE = sc.zeros( 2*(s.Npml+1), 'f')
		sigmaH = sc.zeros( 2*(s.Npml+1), 'f')

		for i in xrange(s.Npml):
			sigmaE[i+1] = pow( (s.Npml-0.5-i)/s.Npml, m )*sigma_max
			sigmaE[i+s.Npml+1] = pow( (0.5+i)/s.Npml, m )*sigma_max
			sigmaH[i+1] = pow( float(s.Npml-i)/s.Npml, m )*sigma_max
			sigmaH[i+s.Npml+2] = pow( (1.+i)/s.Npml, m )*sigma_max

		bE[:] = sc.exp( -(sigmaE[:] + alpha)*s.dt/ep0 )
		bH[:] = sc.exp( -(sigmaH[:] + alpha)*s.dt/ep0 )
		aE[:] = sigmaE[:]/(sigmaE[:]+alpha)*(s.bE[:]-1)
		aH[:] = sigmaH[:]/(sigmaH[:]+alpha)*(s.bH[:]-1)
		
		del sigmaE, sigmaH


	def updateE( s ):
		if 'f' in s.apply_direction[0]:
			cpml_non_kappa.update_x_e( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npmlx, s.forward, 
					s.Ey, s.Ez, s.Hy, s.Hz, s.CEy, s.CEz,
					s.psixEyf, s.psixEzf, s.bEx, s.aEx )

		if 'b' in s.apply_direction[0]:
			cpml_non_kappa.update_x_e( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npmlx, s.backward, 
					s.Ey, s.Ez, s.Hy, s.Hz, s.CEy, s.CEz,
					s.psixEyb, s.psixEzb, s.bEx, s.aEx )


		if 'f' in s.apply_direction[1]:
			cpml_non_kappa.update_y_e( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.forward, 
					s.Ex, s.Ez, s.Hx, s.Hz, s.CEx, s.CEz,
					s.psiyEzf, s.psiyExf, s.bE, s.aE )

		if 'b' in s.apply_direction[1]:
			cpml_non_kappa.update_y_e( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.backward, 
					s.Ex, s.Ez, s.Hx, s.Hz, s.CEx, s.CEz,
					s.psiyEzb, s.psiyExb, s.bE, s.aE )

		if 'f' in s.apply_direction[2]:
			cpml_non_kappa.update_z_e( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.forward, 
					s.Ex, s.Ey, s.Hx, s.Hy, s.CEx, s.CEy,
					s.psizExf, s.psizEyf, s.bE, s.aE )

		if 'b' in s.apply_direction[2]:
			cpml_non_kappa.update_z_e( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.backward, 
					s.Ex, s.Ey, s.Hx, s.Hy, s.CEx, s.CEy,
					s.psizExb, s.psizEyb, s.bE, s.aE )


	def updateH( s ):
		if 'f' in s.apply_direction[0]:
			cpml_non_kappa.update_x_h( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npmlx, s.forward, 
					s.Ey, s.Ez, s.Hy, s.Hz,
					s.psixHyf, s.psixHzf, s.bHx, s.aHx )

		if 'b' in s.apply_direction[0]:
			cpml_non_kappa.update_x_h( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npmlx, s.backward, 
					s.Ey, s.Ez, s.Hy, s.Hz,
					s.psixHyb, s.psixHzb, s.bHx, s.aHx )

		if 'f' in s.apply_direction[1]:
			cpml_non_kappa.update_y_h( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.forward, 
					s.Ex, s.Ez, s.Hx, s.Hz,
					s.psiyHzf, s.psiyHxf, s.bH, s.aH )

		if 'b' in s.apply_direction[1]:
			cpml_non_kappa.update_y_h( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.backward, 
					s.Ex, s.Ez, s.Hx, s.Hz,
					s.psiyHzb, s.psiyHxb, s.bH, s.aH )

		if 'f' in s.apply_direction[2]:
			cpml_non_kappa.update_z_h( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.forward, 
					s.Ex, s.Ey, s.Hx, s.Hy,
					s.psizHxf, s.psizHyf, s.bH, s.aH )

		if 'b' in s.apply_direction[2]:
			cpml_non_kappa.update_z_h( 
					s.Ncore, s.Nx, s.Ny, s.Nz, s.Npml, s.backward, 
					s.Ex, s.Ey, s.Hx, s.Hy,
					s.psizHxb, s.psizHyb, s.bH, s.aH )
