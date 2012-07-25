#!/usr/bin/env python
# _*_ coding: utf-8 _*_ 

from pylab import *

class cfdtd3D_view_geometry:
	def __init__(s, Nx, Ny, Nz):
		s.Nx, s.Ny, s.Nz	=	Nx, Ny, Nz

	def read_geo_file(s, geo_filename):	
		from scipy.io.numpyio import fread

		print 'Reading the line-array from the %s.line.geo...' % geo_filename
		fd	=	open(geo_filename + '.line.geo', 'rb')
		data_line_array	=	fread(fd, 3*(s.Nx+1)*(s.Ny+1)*(s.Nz+1), 'f')	
		data_line_array	=	data_line_array.reshape(3, s.Nx+1, s.Ny+1, s.Nz+1)
		s.line_x	=	data_line_array[0]
		s.line_y	=	data_line_array[1]
		s.line_z	=	data_line_array[2]
		fd.close()

		print 'Reading the area-array from the %s.area.geo...' % geo_filename
		fd	=	open(geo_filename + '.area.geo', 'rb')
		data_area_array	=	fread(fd, 3*(s.Nx+2)*(s.Ny+2)*(s.Nz+2), 'f')	
		data_area_array	=	data_area_array.reshape(3, s.Nx+2, s.Ny+2, s.Nz+2)
		s.area_x	=	data_area_array[0]
		s.area_y	=	data_area_array[1]
		s.area_z	=	data_area_array[2]
		fd.close()

		s.arr	=	{'x':(s.line_x,s.area_x), 'y':(s.line_y,s.area_y), 'z':(s.line_z,s.area_z)}
		s.geo_dirname	=	'%s.gview' % geo_filename

		import os
		try:
			os.mkdir(s.geo_dirname)
		except:
			print 'Remove the exist dir \'%s\'' %s.geo_dirname
			cmd	=	'rm -rf %s' %s.geo_dirname
			os.system(cmd)
			os.mkdir(s.geo_dirname)
		print 'Create the png dir \'%s\'' %s.geo_dirname	


	def draw_line(s, area_direction, line_direction, a):
		AD, LD	=	area_direction, line_direction
		ALD	=	AD+LD	
		Nbc	=	{'zy':(s.Nx,s.Ny), 'zx':(s.Ny,s.Nx), 'yz':(s.Nx,s.Nz), 'yx':(s.Nz,s.Nx), 'xz':(s.Ny,s.Nz), 'xy':(s.Nz,s.Ny)}
		Nb, Nc	=	Nbc[ALD]
		line_arr	=	s.arr[ LD ][0]

		event	=	0
		#print '%s%d, ALD= %s' %(AD,a,ALD)
		for b in xrange(Nb+1):
			coord	=	[]
			for c in xrange(Nc+1):
				line_index	=	{'zy':(b,c,a), 'zx':(c,b,a), 'yz':(b,a,c), 'yx':(c,a,b), 'xz':(a,b,c), 'xy':(a,c,b)}
				line_index_before	=	{'zy':(b,c-1,a), 'zx':(c-1,b,a), 'yz':(b,a,c-1), 'yx':(c-1,a,b), 'xz':(a,b,c-1), 'xy':(a,c-1,b)}
				line_index_after	=	{'zy':(b,c+1,a), 'zx':(c+1,b,a), 'yz':(b,a,c+1), 'yx':(c+1,a,b), 'xz':(a,b,c+1), 'xy':(a,c+1,b)}
				L	=	line_arr[ line_index[ALD] ]
				if L != 1.:
					#print 'line_%s[%s] = %g' %(LD, line_index[ALD],L)
					N	=	Nc+1
					if c == 0:
						#print '\tc==0'
						coord.append( (c+L)/N )
					elif c == N:
						#print '\tc==N'
						coord.append( (c+1-L)/N )
					else:
						before	=	line_arr[ line_index_before[ALD] ]
						after	=	line_arr[ line_index_after[ALD] ]
						if before == 1 and after == 1:
							#print '\tb==0, a==0'
							coord.append( float(c)/N )
							coord.append( (c+1-L)/N )
						elif before == 1:
							#print '\tnb==1'
							coord.append( (c+L)/N )
						elif after == 1:
							#print '\ta==1'
							coord.append( (c+1-L)/N )
					#print '\tcoord=%s' %coord		
			
			if len(coord) != 0:
				event += 1
				#print '(%d,%d) %s' %(b,c,coord)

			while( len(coord)!=0 ):
				c0	=	coord.pop(0)
				c1	=	coord.pop(0)
				if ALD in ('zy','yz','xz'):
					axvline(b,c0,c1,c='k')			
				elif ALD in ('zx','yx','xy'):
					axhline(b,c0,c1,c='k')			

		return event		


	def draw_area_line(s, area_direction):
		AD	=	area_direction
		print 'Draw the area_%s...' % AD
		N	=	{'x':(s.Nx,s.Ny,s.Nz), 'y':(s.Ny,s.Nx,s.Nz), 'z':(s.Nz,s.Nx,s.Ny)}
		Na	=	N[AD][0]
		area_arr =	s.arr[AD][1]

		for a in xrange(Na+1):
			area_slice	=	{'x':(a+1,slice(1,None),slice(1,None)), 'y':(slice(1,None),a+1,slice(1,None)), 'z':(slice(1,None),slice(1,None),a+1)}
			arr	=	area_arr[area_slice[AD]]
			imshow(rot90(arr), cmap=cm.gray, vmin=-1, vmax=1, interpolation='nearest')

			direction	=	['x','y','z']
			direction.pop( direction.index(AD) )
			for LD in direction:
				event	=	s.draw_line(AD, LD, a)

			Nb, Nc	=	N[AD][1:]
			axis([0,Nb+1,0,Nc+1])
			title_str	=	'%s = %d' %(AD,a)
			title(title_str)
			xlabel(direction[0])
			ylabel(direction[1])
			#colorbar()

			if not arr.all() or event != 0:
				if s.save_opt == 'view':
					show()
				elif s.save_opt == 'save':
					save_filename	=	'./%s/%s_%d.png' %(s.geo_dirname, direction[0]+direction[1], a)
					savefig(save_filename)
				clf()

	def view_geometry(s, save_opt):	
		s.save_opt	=	save_opt
		if s.save_opt == 'view':
			ion()
			figure(figsize=(20,10))
		elif s.save_opt == 'save':
			ioff()
			figure(figsize=(20,10))

		s.draw_area_line('z')	
		s.draw_area_line('y')	
		s.draw_area_line('x')	


#------------------------------------------------------------------
if __name__ == '__main__':
	Nx, Ny, Nz	=	120, 74, 60
	S	=	cfdtd3D_view_geometry(Nx, Ny, Nz)

	import sys
	S.read_geo_file(sys.argv[1])
	S.view_geometry('save')	# view, save

