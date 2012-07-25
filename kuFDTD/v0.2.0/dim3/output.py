'''
 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Ki-Hwan Kim (wbkifun@korea.ac.kr)

 Written date : 2009. 7. 13
 last update  :

 Copyright : GNU GPL
'''

from kufdtd.common import *


class Output:
	def __init__( s, dataform, pt1, pt2, spatial_step=(1,1,1) ):
		"""
		dataform : string format of output data 
		pt1 = (pt1x, pt1y, pt1z) : left lower corner
		pt2 = (pt2x, pt2y, pt2z) : right upper corner
		spatial_step = (stepx, stepy, stepz) : data slice step
		"""
		s.dataform = dataform.lower()
		dataform_list = ('e', 'e2', 'ex', 'ey', 'ez', 'ex2', 'ey2', 'ez2',\
				'h', 'h2', 'hx', 'hy', 'hz', 'hx2', 'hy2', 'hz2',\
				's', 's2', 'sx', 'sy', 'sz', 'sx2', 'sy2', 'sz2')
		assert s.dataform in dataform_list, 'The dataform is not one of %s' % str(dataform_list)

		s.spatial_step = spatial_step

		s.pt1 = list( pt1 )
		s.pt2 = list( pt2 )

		s.data_list = []


	def set_space( s, Space ):
		s.Space = Space

		for i in xrange(3):
			if s.pt1[i] == None: s.pt1[i] = 1
			if s.pt2[i] == None: s.pt2[i] = [Space.Nx, Space.Ny, Space.Nz][i]-2
		s.data_slice = [ slice(None), slice(None), slice(None) ]
		for i in xrange(3): 
			if s.pt2[i]-s.pt1[i] == 0: s.data_slice[i] = 0

		s.function_list = s.set_function_list()
		s.E_list, s.H_list  = s.set_field_list()
		s.E_sl_list, s.H_sl_list = s.set_centerize_slice_list()


	def get_data( s ):
		for func in s.function_list: func()
		return s.data_list.pop()[s.data_slice]


	def set_function_list( s ):
		dform = s.dataform
		length_dform = len(dform)
		function_list = [s.make_centerize_data]
		if length_dform == 1:
			function_list.append(s.absolute)
		elif length_dform == 2 and dform[-1] == '2':
			function_list.append(s.square_sum)
		elif length_dform == 3:
			function_list.append(s.square)

		if dform[0] == 's':
			if dform in ['s2', 's']:
				function_list.insert(1, s.poynting)
			else:
				function_list.insert(1, s.ipoynting)
		return tuple(function_list)


	def set_field_list( s ):
		S = s.Space
		dform = s.dataform

		if dform[0] == 'e':
			H_list = []
			if 'x' in dform: E_list = [S.Ex]
			elif 'y' in dform: E_list = [S.Ey]
			elif 'z' in dform: E_list = [S.Ez]
			else: E_list = [ S.Ex, S.Ey, S.Ez ]

		elif dform[0] == 'h':
			E_list = []
			if 'x' in dform: H_list = [S.Hx]
			elif 'y' in dform: H_list = [S.Hy]
			elif 'z' in dform: H_list = [S.Hz]
			else: H_list = [ S.Hx, S.Hy, S.Hz ]

		elif dform[0] == 's':
			if 'x' in dform:
				E_list = [S.Ey, S.Ez]
				H_list = [S.Hz, S.Hy]
			elif 'y' in dform:
				E_list = [S.Ez, S.Ex]
				H_list = [S.Hx, S.Hz]
			elif 'z' in dform:
				E_list = [S.Ex, S.Ey]
				H_list = [S.Hy, S.Hx]
			else:
				E_list = [ S.Ex, S.Ey, S.Ez ]
				H_list = [ S.Hx, S.Hy, S.Hz ]

		return tuple(E_list), tuple(H_list)


	def set_centerize_slice_list( s ):
		dform = s.dataform
		replace = list_replace

		i1, j1, k1 = s.pt1
		i2, j2, k2 = s.pt2

		sl = [ slice(i1,i2+1), slice(j1,j2+1), slice(k1,k2+1) ]
		h_sl = [ slice(i1+1,i2+2), slice(j1+1,j2+2), slice(k1+1,k2+2) ]

		ex_sl = ( sl, replace( sl, 0, slice(i1+1,i2+2) ) )
		ey_sl = ( sl, replace( sl, 1, slice(j1+1,j2+2) ) )
		ez_sl = ( sl, replace( sl, 2, slice(k1+1,k2+2) ) )

		hx_sl = ( sl,
				replace( sl, 1, slice(j1+1,j2+2) ), 
				replace( sl, 2, slice(k1+1,k2+2) ), 
				replace( h_sl, 0, slice(i1,i2+1) ) )
		hy_sl = ( sl, 
				replace( sl, 0, slice(i1+1,i2+2) ),
				replace( sl, 2, slice(k1+1,k2+2) ),
				replace( h_sl, 1, slice(j1,j2+1) ) )
		hz_sl = ( sl, 
				replace( sl, 0, slice(i1+1,i2+2) ),
				replace( sl, 1, slice(j1+1,j2+2) ),
				replace( h_sl, 2, slice(k1,k2+1) ) )

		if dform[0] == 'e':
			H_sl_list = []
			if 'x' in dform: E_sl_list = [ ex_sl ]
			elif 'y' in dform: E_sl_list = [ ey_sl ]
			elif 'z' in dform: E_sl_list = [ ez_sl ]
			else: E_sl_list = [ ex_sl, ey_sl, ez_sl ]

		elif dform[0] == 'h':
			E_sl_list = []
			if 'x' in dform: H_sl_list = [ hx_sl ]
			elif 'y' in dform: H_sl_list =  [ hy_sl ]
			elif 'z' in dform: H_sl_list = [ hz_sl ]
			else: H_sl_list = [ hx_sl, hy_sl, hz_sl ]

		elif dform[0] == 's':
			if 'x' in dform:
				E_sl_list = [ ey_sl, ez_sl ]
				H_sl_list = [ hz_sl, hy_sl ]
			elif 'y' in dform:
				E_sl_list = [ ez_sl, ex_sl ]
				H_sl_list = [ hx_sl, hz_sl ]
			elif 'z' in dform:
				E_sl_list = [ ex_sl, ey_sl ]
				H_sl_list = [ hy_sl, hx_sl ]
			else:
				E_sl_list = [ ex_sl, ey_sl, ez_sl ]
				H_sl_list = [ hx_sl, hy_sl, hz_sl ]

		return tuple(E_sl_list), tuple(H_sl_list)
	

	def make_centerize_data( s ):
		stepx, stepy, stepz = s.spatial_step

		for i, E in enumerate( s.E_list ):
			e_sl = s.E_sl_list[i]
			#print e_sl[0], e_sl[1]
			s.data_list.append( 0.5*( E[ e_sl[0] ] + E[ e_sl[1] ] )\
					[::stepx, ::stepy, ::stepz] )

		for i, H in enumerate( s.H_list ):
			h_sl = s.H_sl_list[i]
			s.data_list.append( 0.25*( H[ h_sl[0] ] + H[ h_sl[1] ] + H[ h_sl[2] ] + H[ h_sl[3] ] )\
					[::stepx, ::stepy, ::stepz]) 
				

	def square( s ):
		for i, dat in enumerate(s.data_list):
			s.data_list[i] = dat**2


	def square_sum( s ):
		s.square()
		sum = 0
		for dat in s.data_list:
			sum = sum + dat
		s.data_list = [sum]


	def absolute( s ):
		s.square_sum()
		s.data_list[0] = sc.sqrt(s.data_list[0])


	def ipoynting( s ):
		e1, e2, h1, h2 = s.data_list
		si = e1*h1 - e2*h2
		s.data_list = [si]

		
	def poynting( s ):
		ex, ey, ez, hx, hy, hz = s.data_list
		sx = ey*hz - ez*hy
		sy = ez*hx - ex*hz
		sz = ex*hy - ey*hx
		s.data_list = [sx, sy, sz]
