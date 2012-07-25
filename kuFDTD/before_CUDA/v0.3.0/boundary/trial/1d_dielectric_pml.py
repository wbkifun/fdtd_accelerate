#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : dielectric.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 10. 29

 Copyright : GNU LGPL

============================== < File Description > ===========================

Temporary code for 1D(Ez,Hy) FDTD

===============================================================================
"""

from time import *
from kufdtd.kufdtd_base import *

#---------------------------------------------
# set parameters
unit_factor = set_unit('Enorm')
Nx = 400
dx = 10e-9
dt = dx/(2*light_velocity)
n1 = 1.
n2 = 2.5

in1 = int(n1)
in2 = int(n2)

dirname = './scbin_files/test-01'


#---------------------------------------------
# PML
def make_gradation_array(number_pml_cells, interval):
	"make a array as symmertry, depth gradation for pml"
	Np = number_pml_cells
	if interval == 'half_depth':
		arr = sc.arange(Np-0.5,-Np-0.5,-1,'f')
		arr[-Np:] = abs(arr[-Np:])
	elif interval == 'full_depth':
		arr = sc.arange(Np,-Np,-1,'f')
		arr[-Np:] = abs(arr[-Np:])+1
	return arr

Np = Npml = 10
kapa_max =  7
alpha = 0.05
m = grade_order = 4
sigma_max = (m+1)/(150*sc.pi*dx)


#---------------------------------------------
# allocate arrays
Ez = sc.zeros(Nx+1,'f')
Hy = sc.zeros(Nx,'f')
ceb = sc.zeros(Nx+1,'f')
epr = sc.ones(Nx+1,'f')

# for abc
abc_f = sc.zeros(2*in1,'f')
abc_b = sc.zeros(2*in2,'f')
abc_f1 = abc_f2 = 0
abc_b1 = abc_b2 = 0

# for pml
psi_ezx = sc.zeros(2*Np, 'f')
psi_hyx = sc.zeros(2*Np, 'f')

sigma_e = make_gradation_array(Np, 'half_depth')
sigma_h = make_gradation_array(Np, 'full_depth')
kapa_e = make_gradation_array(Np, 'half_depth')
kapa_h = make_gradation_array(Np, 'full_depth')

curl_hyx = sc.zeros(Nx-1,'f')
curl_ezx = sc.zeros(Nx,'f')

#---------------------------------------------
# calc pml coefficients
sigma_e[:] = pow(sigma_e[:]/Np, m)*sigma_max
sigma_h[:] = pow(sigma_h[:]/Np, m)*sigma_max 
kapa_e[:] = 1 + (kapa_max - 1)*pow(kapa_e[:]/Np, m)
kapa_h[:] = 1 + (kapa_max - 1)*pow(kapa_h[:]/Np, m)
rcm_b_e = sc.exp( -(sigma_e[:]/kapa_e[:] + alpha)*dt/ep0 )
mrcm_a_e = ( sigma_e[:]/(sigma_e[:]*kapa_e[:] + alpha*(kapa_e[:]**2))*(rcm_b_e[:] - 1) )/dx
rcm_b_h = sc.exp( -(sigma_h[:]/kapa_h[:] + alpha)*dt/ep0 )
mrcm_a_h = ( sigma_h[:]/(sigma_h[:]*kapa_h[:] + alpha*(kapa_h[:]**2))*(rcm_b_h[:] - 1))/dx/unit_factor


#---------------------------------------------
# set geometry
ceb0 = dt/(ep0*dx)*unit_factor
chb0 = dt/(mu0*dx)/unit_factor
#print ceb0, chb0

epr[:] = n1**2
epr[Nx/2:] = n2**2
ceb[:] = ceb0/epr[:]


#---------------------------------------------
# for sin source
wavelength = 1000e-9
wfreq = light_velocity*2*pi/wavelength
period = int( round(2*wavelength/dx) )
print 'period= %g' % period


#---------------------------------------------
# for time average
field_tsum = sc.zeros(Nx+1, 'f')
field2_tsum = sc.zeros(Nx+1, 'f')
field_tavg = sc.zeros(Nx+1, 'f')
field2_tavg = sc.zeros(Nx+1, 'f')
max_tavg_count = 0
last_period = 'off'


#---------------------------------------------
# for plot
from pylab import *
ion()
fig = figure(dpi=150)

line, = plot(ceb)
axis([0,Nx,-1.2,1.2])
#axis([0,Nx,-1e-5,1e-5])
axvline(x=Nx/2, ymin=-1.2, ymax=1.2)
draw()



#---------------------------------------------
# main time loop
#---------------------------------------------
cap_t = 10
TMAX = 10000*period
t0 = time()
for tstep in xrange(TMAX):
	curl_hyx[:] = Hy[1:] - Hy[:-1]
	Ez[1:-1] += ceb[1:-1]*curl_hyx[:]

	# PML
	# front
	psi_ezx[:Np] = rcm_b_e[:Np]*psi_ezx[:Np] + mrcm_a_e[:Np]*curl_hyx[:Np]
	Ez[1:Np+1] += ceb[1:Np+1]*( (1./kapa_e[:Np]-1)*curl_hyx[:Np] + dx*psi_ezx[:Np] )
	# back
	psi_ezx[Np:] = rcm_b_e[Np:]*psi_ezx[Np:] + mrcm_a_e[Np:]*curl_hyx[-Np:]
	Ez[-Np-1:-1] += ceb[-Np-1:-1]*( (1./kapa_e[Np:]-1)*curl_hyx[-Np:] + dx*psi_ezx[Np:] )

	# source
	#pulse = sin(wfreq*dt*tstep)
	pulse = sc.exp( - 0.5*(float(tstep - 200)/30)**2 )
	Ez[Nx/4] += pulse

	'''
	# for ABC
	Ez[0] = abc_f[0]
	for i in xrange(2*in1-1):
		abc_f[i] = abc_f[i+1]
	abc_f[-1] = Ez[1]

	Ez[-1] = abc_b[0]
	for i in xrange(2*in2-1):
		abc_b[i] = abc_b[i+1]
	abc_b[-1] = Ez[-2]
	'''
	
	curl_ezx[:] = Ez[1:] - Ez[:-1]
	Hy[:] += chb0*curl_ezx[:]

	# PML
	# front
	psi_hyx[:Np] = rcm_b_h[:Np]*psi_hyx[:Np] + mrcm_a_h[:Np]*curl_ezx[:Np]
	Hy[:Np] += chb0*( (1./kapa_h[:Np]-1)*curl_ezx[:Np] + unit_factor*dx*psi_hyx[:Np] )
	# back
	psi_hyx[Np:] = rcm_b_h[Np:]*psi_hyx[Np:] + mrcm_a_h[Np:]*curl_ezx[-Np:]
	Hy[-Np:] += chb0*( (1./kapa_h[Np:]-1)*curl_ezx[-Np:] + unit_factor*dx*psi_hyx[Np:] )
		
	if (tstep/cap_t*cap_t == tstep and tstep != 0) or tstep == TMAX-1:
		# for plot
		line.set_ydata(Ez)
		draw()

	'''
	# for time average
	field_tsum[:] += Ez[:]

	if last_period == 'on':
		field2_tsum += Ez[:]**2 

		# write the binary file
		filename = dirname + '/' + 'Ez_%.6d.scbin' % (tstep)
		fd	=	open(filename, 'wb')
		fwrite(fd, Nx, Ez)
		fd.close()

	#---------------------------------------------
	if (tstep/cap_t*cap_t == tstep and tstep != 0) or tstep == TMAX-1:
		t1 = time()
		elapse_time = localtime(t1-t0-60*60*9)
		str_time = strftime('[%j]%H:%M:%S', elapse_time)
		#print '%s    tstep = %d' % (str_time, tstep)
		
		# for plot
		line.set_ydata(Ez)
		draw()

		# for time average
		field_tavg[:] = field_tsum[:]/period
		field_tsum[:] = 0

		max_tavg = abs(field_tavg[:]).max()
		print '%s\ttstep = %d\t(%d period)\t%g' % (str_time, tstep, tstep/period, max_tavg)

		if max_tavg < max_tavg_val:
			max_tavg_count += 1
		else:
			max_tavg_count = 0
		
		# save the field at last period
		if last_period == 'off':
			if max_tavg_count == max_tavg_count_val:
				last_period = 'on'
		elif last_period == 'on':
			field_tavg = field_tsum/period 
			field2_tavg = field2_tsum/period 

			# write the binary file
			filename = dirname + '/' + 'tavg_Ez_%.4dperiod.scbin' % (tstep/period)
			fd	=	open(filename, 'wb')
			fwrite(fd, Nx, field_tavg)
			fd.close()

			filename = dirname + '/' + 'tavg_Ez2_%.4dperiod.scbin' % (tstep/period)
			fd	=	open(filename, 'wb')
			fwrite(fd, Nx, field2_tavg)
			fd.close()

			sys.exit()
	'''

