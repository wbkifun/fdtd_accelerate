'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 2009. 7. 23

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.cpu.base import *
from kufdtd.dim3.cpu.core import dielectric
from kufdtd.dim3.cpu.core import drude_ade

from kufdtd.dim3.structure import is_overlap_3d

class Dielectric( CpuSpace ):
	def __init__( s, Nx, Ny, Nz, dx, Ncore ):
		CpuSpace.__init__( s, Nx, Ny, Nz, dx, Ncore )

		s.shape = (s.Nx, s.Ny, s.Nz)

		s.verify_4xNz( Nz )


	def allocate_main( s ):
		s.Ex = sc.zeros( s.shape, 'f' )
		s.Ey = sc.zeros( s.shape, 'f' )
		s.Ez = sc.zeros( s.shape, 'f' )
		s.Hx = sc.zeros( s.shape, 'f' )
		s.Hy = sc.zeros( s.shape, 'f' )
		s.Hz = sc.zeros( s.shape, 'f' )
		s.mem_usage += 6*s.Ex.nbytes


	def allocate_coeff( s ):
		s.CEx = sc.zeros( s.shape, 'f' )
		s.CEy = sc.zeros( s.shape, 'f' )
		s.CEz = sc.zeros( s.shape, 'f' )
		s.mem_usage += 3*s.CEx.nbytes


	def allocate( s ):
		s.allocate_main()
		s.allocate_coeff()


	def set_coeff( s, structure_groups=[], wrapbox_groups=[] ):
		s.epr_x = sc.ones( s.shape, 'f' )
		s.epr_y = sc.ones( s.shape, 'f' )
		s.epr_z = sc.ones( s.shape, 'f' )

		for i, structure_list in enumerate( structure_groups ):
			c1 = wrapbox_groups[i][2]
			c2 = s.center_pt
			d1 = wrapbox_groups[i][3]
			d2 = s.length
			if is_overlap_3d( c1, c2, d1, d2 ):
				wb_pt1 = list( wrapbox_groups[i][0] )
				wb_pt2 = list( wrapbox_groups[i][1] )
				for i in xrange(3):
					if wb_pt1[i] < s.wrapbox_pt1[i]: wb_pt1[i] = s.wrapbox_pt1[i] 
					if wb_pt2[i] > s.wrapbox_pt2[i]: wb_pt2[i] = s.wrapbox_pt2[i] 
				wb_pt1[0] -= s.wrapbox_pt1[0]
				wb_pt2[0] -= s.wrapbox_pt1[0]

				i1, j1, k1 = sc.int32( sc.array(wb_pt1,'f')/s.dx ) + 1
				i2, j2, k2 = sc.int32( sc.array(wb_pt2,'f')/s.dx ) + 1
				for st in structure_list:
					for i in xrange( i1, i2+1 ):
						for j in xrange( j1, j2+1 ):
							for k in xrange( k1, k2+1 ):
								pt_real_x = ( s.wrapbox_pt1[0] + (i-1)*s.dx, (j-0.5)*s.dx, (k-0.5)*s.dx )
								pt_real_y = ( s.wrapbox_pt1[0] + (i-0.5)*s.dx, (j-1)*s.dx, (k-0.5)*s.dx )
								pt_real_z = ( s.wrapbox_pt1[0] + (i-0.5)*s.dx, (j-0.5)*s.dx, (k-1)*s.dx )
								if st.is_in( pt_real_x ): s.epr_x[i,j,k] = st.matter_parameter[0]
								if st.is_in( pt_real_y ): s.epr_y[i,j,k] = st.matter_parameter[0]
								if st.is_in( pt_real_z ): s.epr_z[i,j,k] = st.matter_parameter[0]

		s.CEx[1:,1:-1,1:-1] = 0.5/s.epr_x[1:,1:-1,1:-1]
		s.CEy[1:-1,1:,1:-1] = 0.5/s.epr_y[1:-1,1:,1:-1]
		s.CEz[1:-1,1:-1,1:] = 0.5/s.epr_z[1:-1,1:-1,1:]

		del s.epr_x, s.epr_y, s.epr_z


	def updateE( s ):
		dielectric.update_e( 
				s.Ncore, s.Nx, s.Ny, s.Nz, 
				s.Ex, s.Ey, s.Ez, 
				s.Hx, s.Hy, s.Hz,
				s.CEx, s.CEy, s.CEz )
	

	def updateH( s ):
		dielectric.update_h( 
				s.Ncore, s.Nx, s.Ny, s.Nz, 
				s.Ex, s.Ey, s.Ez, 
				s.Hx, s.Hy, s.Hz )



class Drude( Dielectric ):
	def __init__( s, Nx, Ny, Nz, dx, Ncore ):
		Dielectric.__init__( s, Nx, Ny, Nz, dx, Ncore )

		s.spaceNyz, s.spaceNz = Ny*Nz, Nz

		s.i0, s.j0, s.k0 = 0, 0, 0
		s.idx0 = s.i0*s.spaceNyz + s.j0*s.spaceNz + s.k0


	def allocate_main( s ):
		s.Jx = sc.zeros( s.shape, 'f' )
		s.Jy = sc.zeros( s.shape, 'f' )
		s.Jz = sc.zeros( s.shape, 'f' )
		s.mem_usage += 3*s.Jx.nbytes


	def allocate_coeff( s ):
		s.CJAx = sc.zeros( s.shape, 'f' )
		s.CJAy = sc.zeros( s.shape, 'f' )
		s.CJAz = sc.zeros( s.shape, 'f' )
		s.CJBx = sc.zeros( s.shape, 'f' )
		s.CJBy = sc.zeros( s.shape, 'f' )
		s.CJBz = sc.zeros( s.shape, 'f' )
		s.mem_usage += 6*s.CJAx.nbytes
		

	def allocate( s ):
		Dielectric.allocate_main( s )
		Dielectric.allocate_coeff( s )

		s.allocate_main()
		s.allocate_coeff()


	def set_coeff( s, structure_groups, wrapbox_groups ):
		s.epr_x = sc.ones( s.shape, 'f' )
		s.epr_y = sc.ones( s.shape, 'f' )
		s.epr_z = sc.ones( s.shape, 'f' )
		s.pfreq_x = sc.zeros( s.shape, 'f' )
		s.pfreq_y = sc.zeros( s.shape, 'f' )
		s.pfreq_z = sc.zeros( s.shape, 'f' )
		s.gamma_x = sc.zeros( s.shape, 'f' )
		s.gamma_y = sc.zeros( s.shape, 'f' )
		s.gamma_z = sc.zeros( s.shape, 'f' )

		for i, structure_list in enumerate( structure_groups ):
			c1 = wrapbox_groups[i][2]
			c2 = s.center_pt
			d1 = wrapbox_groups[i][3]
			d2 = s.length
			if is_overlap_3d( c1, c2, d1, d2 ):
				wb_pt1 = list( wrapbox_groups[i][0] )
				wb_pt2 = list( wrapbox_groups[i][1] )
				for i in xrange(3):
					if wb_pt1[i] < s.wrapbox_pt1[i]: wb_pt1[i] = s.wrapbox_pt1[i] 
					if wb_pt2[i] > s.wrapbox_pt2[i]: wb_pt2[i] = s.wrapbox_pt2[i] 
				wb_pt1[0] -= s.wrapbox_pt1[0]
				wb_pt2[0] -= s.wrapbox_pt1[0]

				i1, j1, k1 = sc.int32( sc.array(wb_pt1,'f')/s.dx ) + 1
				i2, j2, k2 = sc.int32( sc.array(wb_pt2,'f')/s.dx ) + 1
				for st in structure_list:
					for i in xrange( i1, i2+1 ):
						for j in xrange( j1, j2+1 ):
							for k in xrange( k1, k2+1 ):
								pt_real_x = ( s.wrapbox_pt1[0] + (i-1)*s.dx, (j-0.5)*s.dx, (k-0.5)*s.dx )
								pt_real_y = ( s.wrapbox_pt1[0] + (i-0.5)*s.dx, (j-1)*s.dx, (k-0.5)*s.dx )
								pt_real_z = ( s.wrapbox_pt1[0] + (i-0.5)*s.dx, (j-0.5)*s.dx, (k-1)*s.dx )
								if st.is_in( pt_real_x ): 
									s.epr_x[i,j,k] = st.matter_parameter[0]
									s.pfreq_x[i,j,k] = st.matter_parameter[1]
									s.gamma_x[i,j,k] = st.matter_parameter[2]
								if st.is_in( pt_real_y ):
									s.epr_y[i,j,k] = st.matter_parameter[0]
									s.pfreq_y[i,j,k] = st.matter_parameter[1]
									s.gamma_y[i,j,k] = st.matter_parameter[2]
								if st.is_in( pt_real_z ):
									s.epr_z[i,j,k] = st.matter_parameter[0]
									s.pfreq_z[i,j,k] = st.matter_parameter[1]
									s.gamma_z[i,j,k] = st.matter_parameter[2]

		s.CEx[1:,1:-1,1:-1] = 0.5/s.epr_x[1:,1:-1,1:-1]
		s.CEy[1:-1,1:,1:-1] = 0.5/s.epr_y[1:-1,1:,1:-1]
		s.CEz[1:-1,1:-1,1:] = 0.5/s.epr_z[1:-1,1:-1,1:]

		tmp = sc.zeros( s.shape, 'f' )

		tmp[:,:,:] = (2+s.gamma_x[:,:,:]*s.dt)
		s.CJAx[:,:,:] = (2-s.gamma_x[:,:,:]*s.dt)/tmp[:,:,:]
		s.CJBx[:,:,:] = 4*(s.dt*s.pfreq_x[:,:,:])**2/tmp[:,:,:]

		tmp[:,:,:] = (2+s.gamma_y[:,:,:]*s.dt)
		s.CJAy[:,:,:] = (2-s.gamma_y[:,:,:]*s.dt)/tmp[:,:,:]
		s.CJBy[:,:,:] = 4*(s.dt*s.pfreq_y[:,:,:])**2/tmp[:,:,:]

		tmp[:,:,:] = (2+s.gamma_z[:,:,:]*s.dt)
		s.CJAz[:,:,:] = (2-s.gamma_z[:,:,:]*s.dt)/tmp[:,:,:]
		s.CJBz[:,:,:] = 4*(s.dt*s.pfreq_z[:,:,:])**2/tmp[:,:,:]
		
		del s.epr_x, s.epr_y, s.epr_z
		del tmp
		del s.pfreq_x, s.pfreq_y, s.pfreq_z
		del s.gamma_x, s.gamma_y, s.gamma_z


	def updateE( s ):
		Dielectric.updateE( s )
		drude_ade.update_e( 
				s.Ncore, s.Nx, s.Ny, s.Nz,
				s.idx0, s.spaceNyz, s.spaceNz,
				s.Ex, s.Ey, s.Ez,
				s.CEx, s.CEy, s.CEz,
				s.Jx, s.Jy, s.Jz,
				s.CJAx, s.CJAy, s.CJAz,
				s.CJBx, s.CJBy, s.CJBz )



class DrudeAde( CpuSpace ):
	def __init__( s, Space, Nx, Ny, Nz, i0, j0, k0 ):
		CpuSpace.__init__( s, Nx, Ny, Nz, Space.dx, Space.Ncore )
		s.Ex, s.Ey, s.Ez = Space.Ex, Space.Ey, Space.Ez
		s.CEx, s.CEy, s.CEz = Space.CEx, Space.CEy, Space.CEz
		s.spaceNyz, s.spaceNz = Space.Ny*Space.Nz, Space.Nz

		s.i0, s.j0, s.k0 = i0, j0, k0
		s.idx0 = i0*s.spaceNyz + j0*s.spaceNz + k0

		s.shape = (Nx, Ny, Nz)
		s.matter_type = 'drude'


	def allocate_main( s ):
		s.Jx = sc.zeros( s.shape, 'f' )
		s.Jy = sc.zeros( s.shape, 'f' )
		s.Jz = sc.zeros( s.shape, 'f' )
		s.mem_usage += 3*s.Jx.nbytes


	def allocate_coeff( s ):
		s.CJAx = sc.zeros( s.shape, 'f' )
		s.CJAy = sc.zeros( s.shape, 'f' )
		s.CJAz = sc.zeros( s.shape, 'f' )
		s.CJBx = sc.zeros( s.shape, 'f' )
		s.CJBy = sc.zeros( s.shape, 'f' )
		s.CJBz = sc.zeros( s.shape, 'f' )
		s.mem_usage += 6*s.CJAx.nbytes
		
		s.pfreq_x = sc.zeros( s.shape, 'f' )
		s.pfreq_y = sc.zeros( s.shape, 'f' )
		s.pfreq_z = sc.zeros( s.shape, 'f' )
		s.gamma_x = sc.zeros( s.shape, 'f' )
		s.gamma_y = sc.zeros( s.shape, 'f' )
		s.gamma_z = sc.zeros( s.shape, 'f' )


	def allocate( s ):
		s.allocate_main()
		s.allocate_coeff()


	def set_coeff( s, wrapbox_pt1, wrapbox_pt2, structure_list ):
		for st in structure_list:
			for i in xrange( 1, s.Nx ):
				for j in xrange( 1, s.Ny ):
					for k in xrange( 1, s.Nz ):
						pt_real_x = ( wrapbox_pt1[0] + (i-1)*s.dx, 
								wrapbox_pt1[1] + (j-0.5)*s.dx,
								wrapbox_pt1[2] + (k-0.5)*s.dx )
						pt_real_y = ( wrapbox_pt1[0] + (i-0.5)*s.dx, 
								wrapbox_pt1[1] + (j-1)*s.dx, 
								wrapbox_pt1[2] + (k-0.5)*s.dx )
						pt_real_z = ( wrapbox_pt1[0] + (i-0.5)*s.dx, 
								wrapbox_pt1[1] + (j-0.5)*s.dx,
								wrapbox_pt1[2] + (k-1)*s.dx )
						if st.is_in( pt_real_x ): 
							s.pfreq_x[i,j,k] = st.matter_parameter[1]
							s.gamma_x[i,j,k] = st.matter_parameter[2]
						if st.is_in( pt_real_y ): 
							s.pfreq_y[i,j,k] = st.matter_parameter[1]
							s.gamma_y[i,j,k] = st.matter_parameter[2]
						if st.is_in( pt_real_z ):
							s.pfreq_z[i,j,k] = st.matter_parameter[1]
							s.gamma_z[i,j,k] = st.matter_parameter[2]

		tmp = sc.zeros( s.shape, 'f' )

		tmp[:,:,:] = (2+s.gamma_x[:,:,:]*s.dt)
		s.CJAx[:,:,:] = (2-s.gamma_x[:,:,:]*s.dt)/tmp[:,:,:]
		s.CJBx[:,:,:] = 4*(s.dt*s.pfreq_x[:,:,:])**2/tmp[:,:,:]

		tmp[:,:,:] = (2+s.gamma_y[:,:,:]*s.dt)
		s.CJAy[:,:,:] = (2-s.gamma_y[:,:,:]*s.dt)/tmp[:,:,:]
		s.CJBy[:,:,:] = 4*(s.dt*s.pfreq_y[:,:,:])**2/tmp[:,:,:]

		tmp[:,:,:] = (2+s.gamma_z[:,:,:]*s.dt)
		s.CJAz[:,:,:] = (2-s.gamma_z[:,:,:]*s.dt)/tmp[:,:,:]
		s.CJBz[:,:,:] = 4*(s.dt*s.pfreq_z[:,:,:])**2/tmp[:,:,:]
		
		del tmp
		del s.pfreq_x, s.pfreq_y, s.pfreq_z
		del s.gamma_x, s.gamma_y, s.gamma_z


	def updateE( s ):
		drude_ade.update_e( 
				s.Ncore, s.Nx, s.Ny, s.Nz,
				s.idx0, s.spaceNyz, s.spaceNz,
				s.Ex, s.Ey, s.Ez,
				s.CEx, s.CEy, s.CEz,
				s.Jx, s.Jy, s.Jz,
				s.CJAx, s.CJAy, s.CJAz,
				s.CJBx, s.CJBy, s.CJBz )

	

class Drude2( Dielectric ):
	def __init__( s, Nx, Ny, Nz, dx, Ncore ):
		Dielectric.__init__( s, Nx, Ny, Nz, dx, Ncore )


	def allocate_drude_space( s, structure_groups, wrapbox_groups ): 
		s.drude_space_list = []
		for i, structure_list in enumerate( structure_groups ):
			c1 = wrapbox_groups[i][2]
			c2 = s.center_pt
			d1 = wrapbox_groups[i][3]
			d2 = s.length
			if is_overlap_3d( c1, c2, d1, d2 ) and structure_list[0].matter_type == 'drude':
				wb_pt1 = list( wrapbox_groups[i][0] )
				wb_pt2 = list( wrapbox_groups[i][1] )
				for i in xrange(3):
					if wb_pt1[i] < s.wrapbox_pt1[i]: wb_pt1[i] = s.wrapbox_pt1[i] 
					if wb_pt2[i] > s.wrapbox_pt2[i]: wb_pt2[i] = s.wrapbox_pt2[i] 
				local_wb_pt1 = []
				local_wb_pt2 = []
				for i in xrange(3):
					local_wb_pt1.append( wb_pt1[i] )
					local_wb_pt2.append( wb_pt2[i] )
				local_wb_pt1[0] -= s.wrapbox_pt1[0]
				local_wb_pt2[0] -= s.wrapbox_pt1[0]

				i1, j1, k1 = sc.int32( sc.array(local_wb_pt1,'f')/s.dx + 1e-5 ) + 1
				i2, j2, k2 = sc.int32( sc.array(local_wb_pt2,'f')/s.dx + 1e-5 ) + 1

				Nx = i2 - i1
				Ny = j2 - j1
				Nz = k2 - k1
				s.drude_space_list.append( DrudeAde( s, Nx, Ny, Nz, i1, j1, k1 ) )
				s.drude_space_list[-1].allocate()
				s.drude_space_list[-1].set_coeff( wb_pt1, wb_pt2, structure_list )


	def updateE( s ):
		Dielectric.updateE( s )
		for drude_space in s.drude_space_list:
			drude_space.updateE()
