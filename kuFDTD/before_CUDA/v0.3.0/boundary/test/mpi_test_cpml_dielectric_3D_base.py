#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : mpi_test_cpml_dielectric_3D_base.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 7. 11

 Copyright : GNU GPL

============================== < File Description > ===========================

mpi test code, 1 server, 4 node

===============================================================================
"""

from time import *
from pylab import *
from scipy import zeros, sin, pi, exp

from kufdtd.kufdtd_base import *
from kufdtd.boundary.mpi import pypar, myrank, Mpi

# basic variable
dimension = '3D'
ds = 10e-9
grid_opt = 'hfaced'
unit_opt = 'Enorm'
pml_opt = 'xyz'
pbc_opt = ''

# variable for mpi
total_number_cells = (200, 200, 30)
number_mpi_blocks = (2, 2, 1)

number_cells = []
for axis in xrange(3):
	number_cells.append( int(total_number_cells[axis]/number_mpi_blocks[axis]) )
Nx, Ny, Nz = number_cells[0], number_cells[1], number_cells[2]

# variable for PML
number_pml_cells = 10
kapa_max = 7
alpha = 0.05
grade_order = 4

# for graphics using matplotlib
cap_t = 20 # capture_time

if myrank == server:
	ion()
	#figure(figsize=(10,5))
	figure()

	tNx, tNy, tNz = total_number_cells

	intensity = zeros((tNx, tNy), 'f')

	imsh = imshow(
			transpose(intensity),
			origin='lower',
			cmap=cm.hot,
			vmin=0, vmax=0.05,
			interpolation='bilinear')
	colorbar()

	# for data capture
	cap_pt = Nx/2 # capture_point

else:
	from kufdtd.matter.dielectric import Dielectric
	from kufdtd.boundary.cpml import Cpml

	# construct the matter object
	space = Dielectric('3D', ds, number_cells, grid_opt, unit_opt)
	space.allocate_arrays()
	Ex, Ey, Ez = space.efield[0], space.efield[1], space.efield[2]
	Hx, Hy, Hz = space.hfield[0], space.hfield[1], space.hfield[2]
	space.set_coefficients()

	# construct the mpi object
	if dimension == '3D':
		if grid_opt == 'efaced':
			exchange_efields = [Ex, Ey, Ez]
			exchange_hfields = None
		elif grid_opt == 'hfaced':
			exchange_efields = None
			exchange_hfields = [Hx, Hy, Hz]

	elif dimension == '2DTEz':
		if grid_opt == 'efaced':
			exchange_efields = [Ez]
			exchange_hfields = None
		elif grid_opt == 'hfaced':
			exchange_efields = None
			exchange_hfields = [Hx, Hy]

	elif dimension == '2DTHz':
		if grid_opt == 'efaced':
			exchange_efields = [Ex, Ey]
			exchange_hfields = None
		elif grid_opt == 'hfaced':
			exchange_efields = None
			exchange_hfields = [Hz]

	exchange_fields_list = [exchange_efields, exchange_hfields]

	mpi_space = Mpi(number_mpi_blocks, exchange_fields_list, pml_opt, pbc_opt)

	# construct the PML object
	cpml_parameters = (kapa_max, alpha, grade_order)
	pml_space = Cpml(space, number_pml_cells, cpml_parameters)
	pml_space.allocate_pml_arrays()
	pml_apply_opt = mpi_space.mpi_pml_opt # front and back


	# for sin source
	wavelength = 300e-9
	wfreq = light_velocity*2*pi/wavelength

	# for averaged fields
	avgEz = zeros((Nx, Ny), 'f')

#--------------------------------------------------------------------------
# main time loop
#--------------------------------------------------------------------------
t0 = time()
for tstep in xrange(500):
	if myrank == server:
		if tstep/cap_t*cap_t == tstep:
			t1 = time()
			elapse_time = localtime(t1-t0-60*60*9)
			str_time = strftime('[%j]%H:%M:%S', elapse_time)
			print '%s    tstep = %d' % (str_time, tstep)
			
			#clf
			intensity[:Nx,:Ny] = pypar.receive(1)
			intensity[Nx:,:Ny] = pypar.receive(2)
			intensity[:Nx,Ny:] = pypar.receive(3)
			intensity[Nx:,Ny:] = pypar.receive(4)

			imsh.set_array( transpose(intensity) )
			draw()
			#filename = 'mpi_test_%.6d.png' % tstep
			#savefig(filename)

	else:
		space.update_e()
		pml_space.update_cpml_e(pml_apply_opt)
		
		pulse = sin(wfreq*space.dt*tstep)
		if myrank == 1:
			Ez[Nx/3*2, Ny/3*2, :] += pulse
		
		mpi_space.mpi_exchange_efields()


		space.update_h()
		pml_space.update_cpml_h(pml_apply_opt)

		mpi_space.mpi_exchange_hfields()


		if tstep/cap_t*cap_t == tstep:
			avgEz[:,:] = (Ez[:-1,:-1,14] + Ez[1:,:-1,14] + Ez[:-1,1:,14] + Ez[1:,1:,14])/4
			pypar.send(avgEz.copy(), 0)
