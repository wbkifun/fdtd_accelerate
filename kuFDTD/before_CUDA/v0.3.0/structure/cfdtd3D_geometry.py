#!/usr/bin/env python
# _*_ coding: utf-8 _*_

from cfdtd3D_geo_polygon import accuracy, distance

def pt_on_plane_area(plane, pt):
	from scipy import arccos, pi
	angle	=	0
	Npt		=	len(plane)-1
	for i in xrange(Npt):
		a	=	distance(pt,plane[i])
		if i < Npt-1:
			b	=	distance(pt,plane[i+1])
			c	=	distance(plane[i],plane[i+1])
		else:
			b	=	distance(pt,plane[0])
			c	=	distance(plane[i],plane[0])

		angle += arccos( (a*a + b*b - c*c)/(2*a*b) )	

	if abs(angle - 2*pi) < accuracy:	
		return True
	else:
		return False


class cfdtd3D_geometry:
	def __init__(s, Nx, Ny, Nz, obj_set):
		s.Nx, s.Ny, s.Nz	=	Nx, Ny, Nz
		s.obj_set	=	obj_set

		from scipy import ones
		s.line_x	=	ones((Nx+1,Ny+1,Nz+1),'f')
		s.line_y	=	ones((Nx+1,Ny+1,Nz+1),'f')
		s.line_z	=	ones((Nx+1,Ny+1,Nz+1),'f')

		s.area_x	=	ones((Nx+2,Ny+2,Nz+2),'f')
		s.area_y	=	ones((Nx+2,Ny+2,Nz+2),'f')
		s.area_z	=	ones((Nx+2,Ny+2,Nz+2),'f')

	
	def calc_array_line(s, obj, line_direction):
		LD	=	line_direction
		bound_index	=	{\
				'x':(s.bound_y0, s.bound_y1, s.bound_z0, s.bound_z1),\
				'y':(s.bound_x0, s.bound_x1, s.bound_z0, s.bound_z1),\
				'z':(s.bound_x0, s.bound_x1, s.bound_y0, s.bound_y1) }
		bound_m0, bound_m1	=	bound_index[LD][:2]
		bound_n0, bound_n1	=	bound_index[LD][-2:]

		plane_coeff_index	=	{'x':(0,1,2), 'y':(1,0,2), 'z':(2,0,1)}

		print 'line_direction= %s' %LD
		from scipy import sort
		for m in xrange(bound_m0, bound_m1):
			for n in xrange(bound_n0, bound_n1):
				contact_pt	=	[]
				for plane in obj.planes:
					a	=	plane[-1][ plane_coeff_index[LD][0] ]
					if a != 0:
						b	=	plane[-1][ plane_coeff_index[LD][1] ]
						c	=	plane[-1][ plane_coeff_index[LD][2] ]
						d	=	plane[-1][3]
						x	=	-(1./a)*(b*m+c*n+d)

						cpt_index	=	{'x':(x,m,n), 'y':(m,x,n), 'z':(m,n,x)}
						if pt_on_plane_area(plane, cpt_index[LD]):
							contact_pt.append(x)

				contact_pt = list( sort(contact_pt)	)		

				if len(contact_pt) == 2:
					#print '\n----------------------------------------------------------------'
					x0, x1	=	float(contact_pt[0]), float(contact_pt[1])
					i0, i1	=	int(x0), int(x1)
					#print 'LD:%s (%d,%d) %s i(%d,%d)' %(LD,m,n,contact_pt,i0,i1)

					if LD == 'x':
						line_arr1D	=	s.line_x[:,m,n]
					elif LD == 'y':
						line_arr1D	=	s.line_y[m,:,n]
					elif LD == 'z':
						line_arr1D	=	s.line_z[m,n,:]
					
					#print line_arr1D

					if obj.matter == 'PEC':
						line_arr1D[i0+1:i1]	=	0
					elif obj.matter == 'air':
						line_arr1D[i0+1:i1]	=	1.

					#print line_arr1D

					# boundary value at conformal line
					if i0 == i1:
						line_val	=	x1-x0
						if line_val < 0 or line_val > 1:
							print 'illegal line value: %g' %line_val
						if line_val < accuracy:
							pass
						else:
							if obj.matter == 'PEC':
								line_arr1D[i0]	-=	line_val
							elif obj.matter == 'air':
								line_arr1D[i0]	+=	line_val
					else:
						line_val	=	s.calc_line_conformal_val(obj.matter,'head',x0,i0,line_arr1D)
						if line_val < 0 or line_val > 1:
							print 'illegal line value: %g' %line_val
						if line_val < accuracy:
							line_arr1D[i0]	=	0
						else:
							line_arr1D[i0]	=	line_val

						line_val	=	s.calc_line_conformal_val(obj.matter,'tail',x1,i1,line_arr1D)
						if line_val < 0 or line_val > 1:
							print 'illegal line value: %g' %line_val
						if 	line_val < accuracy:
							line_arr1D[i1]	=	0
						else:	
							line_arr1D[i1]	=	line_val

					#print 'after boundary'
					#print line_arr1D

				#elif len(contact_pt) > 2:
				#	print 'Error: There are more 3 contact points at line_%s' % LD


	def calc_line_conformal_val(s,matter,position,x,i,line_arr1D):
		mx_old	=	line_arr1D[i]	

		# line value
		if matter == 'PEC':
			if position == 'head':
				mx	=	x-i
			elif position == 'tail':
				mx	=	1-(x-i)
		elif matter == 'air':
			if position == 'head':
				mx	=	1-(x-i)
			elif position == 'tail':
				mx	=	x-i

		#print 'mx_old= %g, mx= %g' %(mx_old, mx)
		#print 'position= %s' %(position)

		# position of old line value
		if mx_old != 0 and mx_old != 1:
			mx_old0, mx_old1	=	line_arr1D[i-1], line_arr1D[i+1]
			diff	=	mx_old0 - mx_old1
			if diff == 0 or diff == 1 or (mx_old0 > 0 and mx_old0 < 1):
				position_old	=	'head'
			if diff == -1 or (mx_old1 > 0 and mx_old1 < 1):
				position_old	=	'tail'

			#print 'position_old= %s' %(position_old)

		# line conformal value
		if matter == 'PEC':
			if mx_old == 0:
				line_val	=	0
			elif mx_old == 1:
				line_val	=	mx
			else:
				if (position_old == 'head' and position == 'head') or (position_old == 'tail' and position == 'tail'):
					if mx_old < mx:
						line_val	=	mx_old
					elif mx_old >= mx:
						line_val	=	mx
				elif (position_old == 'head' and position == 'tail') or (position_old == 'tail' and position == 'head'):
					if mx_old + mx < 1:
						line_val	=	0
					elif mx_old + mx >= 1:
						line_val	=	mx

		elif matter == 'air':
			if mx_old == 0:
				line_val	=	mx
			elif mx_old == 1:
				line_val	=	1.
			else:
				if (position_old == 'head' and position == 'head') or (position_old == 'tail' and position == 'tail'):
					if mx_old + mx < 1:
						line_val	=	mx_old + mx
					elif mx_old + mx >= 1:
						line_val	=	1
				elif (position_old == 'head' and position == 'tail') or (position_old == 'tail' and position == 'head'):
					if mx_old < mx:
						line_val	=	mx
					elif mx_old >= mx:
						line_val	=	mx_old

		return line_val				
			

	def calc_array_area(s, area_bound):
		AB	=	area_bound
		for i in xrange(AB[0],AB[1]):
			for j in xrange(AB[2],AB[3]):
				for k in xrange(AB[4],AB[5]):
					s.area_x[i,j,k]	=	s.calc_line2area('x',i,j,k)
					s.area_y[i,j,k]	=	s.calc_line2area('y',i,j,k)
					s.area_z[i,j,k]	=	s.calc_line2area('z',i,j,k)


	def calc_line2area(s, direction, i,j,k):
		if direction	==	'x':
			a0	=	s.line_y[i-1,j-1,k-1]
			a1	=	s.line_y[i-1,j-1,k  ]
			b0	=	s.line_z[i-1,j-1,k-1]
			b1	=	s.line_z[i-1,j  ,k-1]
		elif direction	==	'y':
			a0	=	s.line_x[i-1,j-1,k-1]
			a1	=	s.line_x[i-1,j-1,k  ]
			b0	=	s.line_z[i-1,j-1,k-1]
			b1	=	s.line_z[i  ,j-1,k-1]
		elif direction	==	'z':
			a0	=	s.line_x[i-1,j-1,k-1]
			a1	=	s.line_x[i-1,j  ,k-1]
			b0	=	s.line_y[i-1,j-1,k-1]
			b1	=	s.line_y[i  ,j-1,k-1]

		line	=	[a0,a1,b0,b1]
		#print '\n(AD=%s, (%d,%d,%d) line=%s' %(direction,i,j,k,line)
		if line.count(0) > 2:
			#print 'count(0) > 2'
			result	=	0

		elif line.count(1.) > 2:
			#print 'count(1) > 2'
			result	=	1

		elif line.count(0) == 2:
			#print 'count(0) == 2'
			non_zeros	=	[]
			sum			=	0
			for i in xrange( len(line) ):
				if line[i] != 0:
					sum += i
					non_zeros.append(line[i])

			#print 'line_index_sum= %d, non_zeors_line=%s' %(sum,non_zeros)
			if sum == 5 or sum == 1:
				result	=	0.5*(non_zeros[0] + non_zeros[1])
			else:
				result	=	0.5*(non_zeros[0]*non_zeros[1])

		elif line.count(1) == 2:
			#print 'count(1) == 2'
			non_ones	=	[]
			sum		=	0
			for i in xrange( len(line) ):
				if line[i] != 1:
					sum += i
					non_ones.append(line[i])

			#print 'line_index_sum= %d, non_ones_line=%s' %(sum,non_ones)
			if sum == 5 or sum == 1:
				result	=	1 - 0.5*((1-non_ones[0]) + (1-non_ones[1]))
			else:
				result	=	1 - 0.5*((1-non_ones[0])*(1-non_ones[1]))

		elif line.count(0) == 1:
			#print 'count(0) == 1'
			i	=	line.index(0)
			if i == 0 or i == 1:
				result	=	0.5*(line[2] + line[3])
			else:
				result	=	0.5*(line[0] + line[1])

		elif line.count(1) == 1:
			#print 'count(1) == 1'
			i	=	line.index(1)
			if i == 0 or i == 1:
				result	=	1 - 0.5*((1-line[2]) + (1-line[3]))
			else:
				result	=	1 - 0.5*((1-line[0]) + (1-line[1]))

		#print 'result = %g' %(result)
		if result	<	accuracy**2:
			result	=	0
		return result


	def make_geometry_array(s):
		area_bound_x	=	[]
		area_bound_y	=	[]
		area_bound_z	=	[]
		for obj in s.obj_set:
			s.bound_x0, s.bound_x1	=	obj.bound_x[0], obj.bound_x[1]
			s.bound_y0, s.bound_y1	=	obj.bound_y[0], obj.bound_y[1]
			s.bound_z0, s.bound_z1	=	obj.bound_z[0], obj.bound_z[1]
			area_bound_x.append(s.bound_x0)
			area_bound_x.append(s.bound_x1)
			area_bound_y.append(s.bound_y0)
			area_bound_y.append(s.bound_y1)
			area_bound_z.append(s.bound_z0)
			area_bound_z.append(s.bound_z1)
			#print '(%d,%d), (%d,%d)' %(s.bound_x0, s.bound_x1, s.bound_z0, s.bound_z1)

			print 'Calculate the line-array for the conformal geometry...'
			s.calc_array_line(obj,'x')
			s.calc_array_line(obj,'y')
			s.calc_array_line(obj,'z')

		print 'Calculate the area-array from the line...'
		area_bound	=	[min(area_bound_x)+1, max(area_bound_x)+1,\
						min(area_bound_y)+1, max(area_bound_y)+1,\
						min(area_bound_z)+1, max(area_bound_z)+1]
		s.calc_array_area(area_bound)


	def save_geometry_array(s, save_filename):
		print 'Save the geo-file as \'%s\'...' % save_filename
		save_filename_line	=	save_filename.replace('.geo','.line.geo')
		save_filename_area	=	save_filename.replace('.geo','.area.geo')

		from scipy import array
		data_line = (s.line_x, s.line_y, s.line_z)
		data_area = (s.area_x, s.area_y, s.area_z)
		data_line_array = array(data_line)
		data_area_array = array(data_area)

		from scipy.io.numpyio import fwrite
		fd	=	open(save_filename_line,'wb')
		fwrite(fd, data_line_array.size, data_line_array)
		fd.close()
		fd	=	open(save_filename_area,'wb')
		fwrite(fd, data_area_array.size, data_area_array)
		fd.close()
