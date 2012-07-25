'''
 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Ki-Hwan Kim (wbkifun@korea.ac.kr)

 Written date : 2009. 7. 16
 last update  : 

 Copyright : GNU GPL
'''

from kufdtd.common import *


class Fdtd1d:
	def __init__( s, Nx, dt, wfreq, ratio_pv ):
		s.dt = dt
		s.wfreq = wfreq
		s.ratio_pv = ratio_pv

		s.E = sc.zeros( Nx+10, 'f' )
		s.H = sc.zeros( Nx+10, 'f' )
		s.abc_f1, s.abc_f2 = 0, 0
		s.abc_b1, s.abc_b2 = 0, 0

	
	def updateE( s, tstep ):
		s.E[1:-1] -= 0.5*s.ratio_pv*( s.H[2:] - s.H[1:-1] ) 

		# abc
		s.E[0] = s.abc_f1
		s.abc_f1 = s.abc_f2
		s.abc_f2 = s.E[1]

		s.E[-1] = s.abc_b1
		s.abc_b1 = s.abc_b2
		s.abc_b2 = s.E[-2]

		s.E[3] = sc.sin( s.wfreq*s.dt*tstep )


	def updateH( s ):
		s.H[1:] -= 0.5*s.ratio_pv*( s.E[1:] - s.E[:-1] )

		

class TfsfIncNormal:
	def __init__( s, space1d, ce1, ce2, ch1, ch2, sliceE1, sliceE2, sliceH1, sliceH2 ):
		s.space1d = space1d
		s.ce1, s.ce2 = ce1, ce2
		s.ch1, s.ch2 = ch1, ch2
		s.sliceE1, s.sliceE2 = sliceE1, sliceE2
		s.sliceH1, s.sliceH2 = sliceH1, sliceH2

	def set_newaxis_i( s, newaxis_i ):
		s.sl_expand = [slice(None)]
		s.sl_expand.insert( newaxis_i, sc.newaxis )

	def getE( s ):
		inc1 = s.ce1*s.space1d.E[s.sliceE1]
		inc2 = s.ce2*s.space1d.E[s.sliceE2]
		return inc1[s.sl_expand], inc2[s.sl_expand]

	def getH( s ):
		inc1 = s.ch1*s.space1d.H[s.sliceH1]
		inc2 = s.ch2*s.space1d.H[s.sliceH2]
		return inc1[s.sl_expand], inc2[s.sl_expand] 



class TfsfFace:
	def __init__( s, e1, e2, h1, h2, slice_e1, slice_e2, slice_h1, slice_h2 ):
		s.e1, s.e2 = e1, e2
		s.h1, s.h2 = h1, h2
		s.slice_e1, s.slice_e2 = slice_e1, slice_e2
		s.slice_h1, s.slice_h2 = slice_h1, slice_h2
		#print 'slice:', s.slice_e1, s.slice_e2, s.slice_h1, s.slice_h2


	def updateE( s, Inc ):
		h1_inc, h2_inc = Inc.getH()
		#print 'shape: ', s.e1[s.slice_e1].shape, h1_inc.shape
		s.e1[s.slice_e1] += 0.5*h1_inc
		s.e2[s.slice_e2] += 0.5*h2_inc


	def updateH( s, Inc ):
		e1_inc, e2_inc = Inc.getE()
		s.h1[s.slice_h1] += 0.5*e1_inc
		s.h2[s.slice_h2] += 0.5*e2_inc



class Tfsf:
	def __init__( s, pt1, pt2, apply_direction, wavelength, propagation_direction, polarization_angle ):
		s.pt1, s.pt2 = pt1, pt2
		s.apply_direction = apply_direction
		s.wavelength = wavelength
		s.propagation_direction = propagation_direction
		s.p_angle = polarization_angle

	
	def set_space( s, Space ):
		s.Space = Space
		pt1 = list( s.pt1 )
		pt2 = list( s.pt2 )
		for i in xrange(3):
			if pt1[i] == None: pt1[i] = 1
			if pt2[i] == None: pt2[i] = [Space.Nx, Space.Ny, Space.Nz][i]-2

		for i in xrange(3): pt2[i] += 1

		s.space1d = s.make_space1d( pt1, pt2 )
		s.face_list = s.calc_face_list( pt1, pt2 )

		axis = s.propagation_direction[1]
		if 'x' is axis:   i2 = pt2[0] - pt1[0] + 5
		elif 'y' is axis: i2 = pt2[1] - pt1[1] + 5
		elif 'z' is axis: i2 = pt2[2] - pt1[2] + 5
		s.inc_list = s.calc_inc_list( 5, i2 )


	def make_space1d( s, pt1, pt2 ):
		if s.propagation_direction[0] is 'normal':
			axis = s.propagation_direction[1]
			if 'x' is axis: N = pt2[0] - pt1[0] + 1
			elif 'y' is axis: N = pt2[1] - pt1[1] + 1
			elif 'z' is axis: N = pt2[2] - pt1[2] + 1

			ratio_pv = 1

		elif s.propagation_direction[0] is 'oblique':
			theta = s.propagation_direction[1]
			phi = s.propagation_direction[2]

		wfreq = 2*sc.pi*light_velocity/s.wavelength

		return Fdtd1d( N, s.Space.dt, wfreq, ratio_pv )


	def calc_face_list( s, pt1, pt2 ):
		S = s.Space
		i1, j1, k1 = pt1
		i2, j2, k2 = pt2

		slx  = slice( i1, i2   )
		slxp = slice( i1, i2+1 )
		sly  = slice( j1, j2   )
		slyp = slice( j1, j2+1 )
		slz  = slice( k1, k2   )
		slzp = slice( k1, k2+1 )

		face_list = []

		if 'f' in s.apply_direction[0]:
			face_list.append( TfsfFace( S.Ey, S.Ez, S.Hy, S.Hz,
					(i1-1, slyp, slz), (i1-1, sly, slzp),
					(i1, sly, slzp), (i1, slyp, slz) ) )

		if 'b' in s.apply_direction[0]:
			face_list.append( TfsfFace( S.Ey, S.Ez, S.Hy, S.Hz,
					(i2, slyp, slz), (i2, sly, slzp),
					(i2, sly, slzp), (i2, slyp, slz) ) )

		if 'f' in s.apply_direction[1]:
			face_list.append( TfsfFace( S.Ez, S.Ex, S.Hz, S.Hx, 
					(slx, j1-1, slzp), (slxp, j1-1, slz), 
					(slxp, j1, slz), (slx, j1, slzp) ) )

		if 'b' in s.apply_direction[1]:
			face_list.append( TfsfFace( S.Ez, S.Ex, S.Hz, S.Hx, 
					(slx, j2, slzp), (slxp, j2, slz), 
					(slxp, j2, slz), (slx, j2, slzp) ) )

		if 'f' in s.apply_direction[2]:
			face_list.append( TfsfFace( S.Ex, S.Ey, S.Hx, S.Hy,
					(slxp, sly, k1-1), (slx, slyp, k1-1), 
					(slx, slyp, k1), (slxp, sly, k1) ) )

		if 'b' in s.apply_direction[2]:
			face_list.append( TfsfFace( S.Ex, S.Ey, S.Hx, S.Hy, 
					(slxp, sly, k2), (slx, slyp, k2), 
					(slx, slyp, k2), (slxp, sly, k2) ) )

		return face_list

	
	def calc_inc_list( s, i1, i2 ):
		inc_list = []

		if s.propagation_direction[0] is 'normal':
			axis = s.propagation_direction[1]
			si, co = sc.sin( s.p_angle ), sc.cos( s.p_angle )
			sl1 = slice( i1-1, i1   )
			sl2 = slice( i1  , i1+1 )
			sl3 = slice( i1  , i2   )
			sl4 = slice( i1  , i2+1 )
			sl5 = slice( i2  , i2+1 )

			Inc1 = TfsfIncNormal( s.space1d, -si,  co,  co,  si, sl1, sl1, sl2, sl2 ) 
			Inc2 = TfsfIncNormal( s.space1d,  si, -co, -co, -si, sl5, sl5, sl5, sl5 )
			Inc3 = TfsfIncNormal( s.space1d,   0,  si,   0, -co, sl4, sl3, sl3, sl4 )
			Inc4 = TfsfIncNormal( s.space1d,   0, -si,   0,  co, sl4, sl3, sl3, sl4 )
			Inc5 = TfsfIncNormal( s.space1d, -co,   0, -si,   0, sl3, sl4, sl4, sl3 )
			Inc6 = TfsfIncNormal( s.space1d,  co,   0,  si,   0, sl3, sl4, sl4, sl3 )

			if 'x' is axis: 
				inc_all_list = [Inc1, Inc2, Inc3, Inc4, Inc5, Inc6]
				newaxis_i_list = [ 0, 0, 1, 1, 1, 1 ]
			elif 'y' is axis: 
				inc_all_list = [Inc5, Inc6, Inc1, Inc2, Inc3, Inc4]
				newaxis_i_list = [ 1, 1, 0, 0, 0, 0 ]
			elif 'z' is axis: 
				inc_all_list = [Inc3, Inc4, Inc5, Inc6, Inc1, Inc2]
				newaxis_i_list = [ 0, 0, 0, 0, 0, 0 ]

			for i, inc in enumerate( inc_all_list ):
				inc.set_newaxis_i( newaxis_i_list[i] )
				
			apply_list = []
			for ax in s.apply_direction:
				if 'f' in ax: apply_list.append(1)
				else: apply_list.append(0)
				if 'b' in ax: apply_list.append(1)
				else: apply_list.append(0)
			
			for i, apply in enumerate( apply_list ):
				if apply == 1: inc_list.append( inc_all_list[i] )

		elif s.propagation_direction[0] is 'oblique':
			pass

		return inc_list


	def updateE( s, tstep ):
		s.space1d.updateE( tstep )
		for i, face in enumerate( s.face_list ): 
			face.updateE( s.inc_list[i] )
		

	def updateH( s ):
		s.space1d.updateH()
		for i, face in enumerate( s.face_list ): 
			face.updateH( s.inc_list[i] )
