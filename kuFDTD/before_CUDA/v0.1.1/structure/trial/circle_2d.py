#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : circle_2d.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 7. 2

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the structure classes.
	Circle2D

===============================================================================
"""

from kufdtd.structure.structure_base_2d import *

import numpy

class Circle2D:
	def __init__(self, o_pt, r, epr):
		self.a = o_pt[x_axis]
		self.b = o_pt[y_axis]
		self.r = r
		self.epr = epr


	def calc_intersection_points(self, point, grid_direction):
		a, b, r = self.a, self.b, self.r

		if grid_direction == x_axis:
			y = point

			cterm = sc.sqrt(r**2 - (y-b)**2)
			if type(cterm) is not numpy.complex128:
				x1 = a - cterm
				x2 = a + cterm

				unit_grad_vec1 = ( (x1-a)/r, (y-b)/r )
				unit_grad_vec2 = ( (x2-a)/r, (y-b)/r )
				theta1 = sc.arccos( rotate(unit_grad_vec1, -0.5*pi)[0] )
				theta2 = sc.arccos( rotate(unit_grad_vec2, 0.5*pi)[0] )

				return x1, x2, theta1, theta2

			else:
				return None


		elif grid_direction == y_axis:
			x = point

			cterm = sc.sqrt(r**2 - (x-a)**2)
			if type(cterm) is not numpy.complex128:
				#print cterm
				#print type(cterm)
				y1 = b - cterm
				y2 = b + cterm
				#print 'x=%g, y1=%g, y2=%g' % (x, y1, y2)

				unit_grad_vec1 = ( (x-a)/r, (y1-b)/r )
				unit_grad_vec2 = ( (x-a)/r, (y2-b)/r )
				theta1 = sc.arccos( rotate(unit_grad_vec1, -0.5*pi)[0] )
				theta2 = sc.arccos( rotate(unit_grad_vec2, 0.5*pi)[0] )

				return y1, y2, theta1, theta2

			else:
				return None



"""

Ox, Oy = int(coord_origin[0]/ds), int(coord_origin[1]/ds)

# for circle (x-a)**2 + (y-b)**2 = r**2
a, b, r = 0, 0, 155e-9
background = air = 1	
glass = 1.5	

#x = 0
#cterm = sc.sqrt(r**2 - (x-a)**2)
#y1 = b - cterm
#y2 = b + cterm
#print 'x=%g, y1=%g, y2=%g' % (x, y1, y2)

if space.dimension == '2DHz':
	epr_eff = space.matter_line_arrays

	#print type(Nx)
	for i in xrange(Nx+1):
		x = (i-Ox)*ds

#-----------------------------------------------------------------------------
# set the basic parameters
#-----------------------------------------------------------------------------
dimension = '2DHz'
total_space = (700e-9, 700e-9)
coord_origin = (350e-9, 350e-9)
ds = 10e-9
grid_opt = 'hfaced'
unit_opt = 'Enorm'

#-----------------------------------------------------------------------------
number_cells = (int(total_space[0]/ds), int(total_space[1]/ds))
space = Dielectric('2DHz', ds, number_cells, grid_opt, unit_opt)
space.allocate_arrays()
Nx, Ny = number_cells[0], number_cells[1]
Hz = space.hfield[2]

#-----------------------------------------------------------------------------
# construct the PML object
#-----------------------------------------------------------------------------
number_pml_cells = 10
kapa_max = 7
alpha = 0.05
grade_order = 4
cpml_parameters = (kapa_max, alpha, grade_order)
pml_space = Cpml(space, number_pml_cells, cpml_parameters)
pml_space.allocate_pml_arrays()
pml_apply_opt = ('fb', 'fb') # front and back

#-----------------------------------------------------------------------------
# set epr coefficients
#-----------------------------------------------------------------------------
#print epr_eff
#print space.matter_line_arrays
#print 'len(space.epr)', len(space.epr)
#print 'Nx=%d, Ny=%d' % (Nx, Ny)
#print 'space.efield[0]', space.efield[0].shape
#print 'space.epr[0].shape', space.epr[0].shape

#epr_eff[0][:,:] = 1
#epr_eff[1][:,:] = 1
space.set_coefficients()


#-----------------------------------------------------------------------------
# for graphics using matplotlib
#-----------------------------------------------------------------------------
ion()
#figure(figsize=(10,5))
fig = figure(figsize=(18,12))

'''
# for circle
from matplotlib.patches import Circle
c1 = Circle(xy=(2*Ox,2*Oy), radius=2*r/ds)

ax = fig.add_subplot(111,aspect='equal')
ax.add_artist(c1)
c1.set_clip_box(ax.bbox)
#c1.set_alpha(0.5)
#c1.set_facecolor((1,1,1))
#c1.set_edgecolor((0,0,0))
c1.set_linewidth(1)
c1.set_fill(False)

# for tf/sf
from matplotlib.patches import Rectangle
rect1 = Rectangle(xy=(2*15,2*15), width=2*40, height=2*40)

ax.add_artist(rect1)
rect1.set_clip_box(ax.bbox)
rect1.set_linewidth(1)
rect1.set_fill(False)

# for pml
pml1 = Rectangle(xy=(0,0), width=2*10, height=2*70)
pml2 = Rectangle(xy=(2*60,0), width=2*10, height=2*70)
pml3 = Rectangle(xy=(0,0), width=2*70, height=2*10)
pml4 = Rectangle(xy=(0,2*60), width=2*70, height=2*10)

ax.add_artist(pml1)
ax.add_artist(pml2)
ax.add_artist(pml3)
ax.add_artist(pml4)
pml1.set_clip_box(ax.bbox)
pml2.set_clip_box(ax.bbox)
pml3.set_clip_box(ax.bbox)
pml4.set_clip_box(ax.bbox)

#pml1.set_hatch('/')
pml1.set_facecolor((0,0,0))
pml2.set_facecolor((0,0,0))
pml3.set_facecolor((0,0,0))
pml4.set_facecolor((0,0,0))
pml1.set_alpha(0.3)
pml2.set_alpha(0.3)
pml3.set_alpha(0.3)
pml4.set_alpha(0.3)

# for epr
epr_x = space.epr[0]
epr_y = space.epr[1]
yee_grid = zeros((2*(Nx+1), 2*(Ny+1)), 'f')
for i in xrange(Nx+1):
	for j in xrange(Ny+1):
		yee_grid[2*i+1,2*j] = epr_x[i,j]
		yee_grid[2*i,2*j+1] = epr_y[i,j] 

imsh = imshow(
		yee_grid,
		cmap=cm.hot_r,
		origin='lower',
		vmin=1, vmax=2,
		#interpolation='bilinear')
		interpolation='nearest')
colorbar()

xloc, xlabel = xticks( arange(0,2*(Nx+1),2), arange(Nx+1)) 
yloc, ylabel = yticks( arange(0,2*(Ny+1),2), arange(Ny+1)) 
#print xloc, xlabel
#print type(xloc), type(xlabel)
grid()
show()
'''

from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
ax = fig.add_subplot(111,aspect='equal')

# for pml
pml1 = Rectangle(xy=(0,0), width=10, height=70)
pml2 = Rectangle(xy=(60,0), width=10, height=70)
pml3 = Rectangle(xy=(0,0), width=70, height=10)
pml4 = Rectangle(xy=(0,60), width=70, height=10)

ax.add_artist(pml1)
ax.add_artist(pml2)
ax.add_artist(pml3)
ax.add_artist(pml4)
pml1.set_clip_box(ax.bbox)
pml2.set_clip_box(ax.bbox)
pml3.set_clip_box(ax.bbox)
pml4.set_clip_box(ax.bbox)

#pml1.set_hatch('/')
pml1.set_facecolor((0,0,0))
pml2.set_facecolor((0,0,0))
pml3.set_facecolor((0,0,0))
pml4.set_facecolor((0,0,0))
pml1.set_alpha(0.3)
pml2.set_alpha(0.3)
pml3.set_alpha(0.3)
pml4.set_alpha(0.3)

# for circle
c1 = Circle(xy=(Ox,Oy), radius=r/ds)

ax.add_artist(c1)
c1.set_clip_box(ax.bbox)
#c1.set_alpha(0.5)
#c1.set_facecolor((1,1,1))
c1.set_edgecolor((1,1,1))
c1.set_linewidth(1)
c1.set_fill(False)

imsh = imshow(
		rot90(Hz**2),
		cmap=cm.hot,
		origin='lower',
		vmin=0, vmax=1,
		interpolation='bilinear')
		#interpolation='nearest')
colorbar()

#-----------------------------------------------------------------------------
# for sin source
#-----------------------------------------------------------------------------
wavelength = 400e-9
wfreq = light_velocity*2*pi/wavelength

period = 2*wavelength/ds
print 'period=%g' % period
intensity_sum = zeros((Nx+2,Ny+2),'f')
#print Hz.shape
#print intensity_sum.shape
xsum_tavgI = 0
pavg_xsum_tavgI = 0

#-----------------------------------------------------------------------------
# for data capture
#-----------------------------------------------------------------------------
cap_t = int(period) # capture_time
cap_pt = Nx/2 # capture_point

#-----------------------------------------------------------------------------
# main time loop
#-----------------------------------------------------------------------------
t0 = time()
for tstep in xrange(1000000):
	space.update_e()
	pml_space.update_cpml_e(pml_apply_opt)
	
	space.update_h()
	pml_space.update_cpml_h(pml_apply_opt)
	
	pulse = sin(wfreq*space.dt*tstep)
	space.hfield[2][15,1:-1] += pulse
	
	intensity_sum += Hz[:,:]**2
	#print 'sum', intensity_sum.sum()

	if tstep/cap_t*cap_t == tstep and tstep != 0:
		t1 = time()
		elapse_time = localtime(t1-t0-60*60*9)
		str_time = strftime('[%j]%H:%M:%S', elapse_time)
		print '%s    tstep = %d' % (str_time, tstep)
		
		tavgI = intensity_sum/period

		#clf
		imsh.set_array(rot90(tavgI))
		#imsh.set_array(rot90(Hz[:,:]**2))
		draw()
		
		#print '\t\tdifference1= %g' % abs(pavg_xsum_tavgI-tavgI.sum())
		before = pavg_xsum_tavgI

		xsum_tavgI += tavgI.sum()
		pavg_xsum_tavgI = xsum_tavgI/(tstep/int(period))

		#print '\t\tpavg_xsum_tavgI= %g' % pavg_xsum_tavgI

		difference2 = abs( (pavg_xsum_tavgI-before)/before )*100	# error percentage
		print '\t\tdifference2= %g' % difference2
		
		if difference2 < 0.01: 
			sys.exit()

		intensity_sum[:,:] = 0
"""
