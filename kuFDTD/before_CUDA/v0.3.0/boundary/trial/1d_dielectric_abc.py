#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : dielectric.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 10. 10

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
n1 = 1
n2 = 2

dirname = './scbin_files/test-01'

#---------------------------------------------
# allocate arrays
Ez = sc.zeros(Nx+1,'f')
Hy = sc.zeros(Nx,'f')
ceb = sc.zeros(Nx+1,'f')
epr = sc.ones(Nx+1,'f')
abc_f = sc.zeros(2*n1,'f')
abc_b = sc.zeros(2*n2,'f')

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
#axis([0,Nx,-.1,0.1])
axvline(x=Nx/2, ymin=-1.2, ymax=1.2)
draw()

abc_f1 = abc_f2 = 0
abc_b1 = abc_b2 = 0


#---------------------------------------------
# main time loop
#---------------------------------------------
cap_t = period
TMAX = 10000*period
t0 = time()
for tstep in xrange(TMAX):
	Ez[1:-1] += ceb[1:-1]*(Hy[1:] - Hy[:-1])

	#pulse = sin(wfreq*dt*tstep)
	pulse = sc.exp( - 0.5*(float(tstep - 200)/30)**2 )
	Ez[Nx/4] += pulse

	# for ABC
	Ez[0] = abc_f[0]
	for i in xrange(2*n1-1):
		abc_f[i] = abc_f[i+1]
	abc_f[-1] = Ez[1]

	Ez[-1] = abc_b[0]
	for i in xrange(2*n2-1):
		abc_b[i] = abc_b[i+1]
	abc_b[-1] = Ez[-2]
	
	Hy[:] += chb0*(Ez[1:] - Ez[:-1])

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

