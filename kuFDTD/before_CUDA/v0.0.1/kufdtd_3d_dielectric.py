#!/usr/bin/env python
# _*_ coding: utf-8 _*_ 

"""
 <File Description>

 File Name : kufdtd_3d_dielectric.py

 Author : Kim, Ki-hwan (wbkifun@korea.ac.kr) 
          Kim, KyoungHo (rain_woo@korea.ac.kr)
 Written date : 2008. 01. 02. Wed

 Copyright : This has used lots of python modules which is opend to public. So,
 it is also in pulic.

============================== < File Description > ===========================

이 파일은 KUFDTD(Korea University Finite Difference Time Domain method)의
기본을 이루는 물리적인 변수, 수학적인 정의, 공간적인 정의, 전산모사에서
필요로 하는 pre-defined 변수, 그리고 KUFDTD에서 유용하게 쓰일 파이썬 함수들을
선언한다.

===============================================================================
"""
import sys
sys.path.append('./')
from kufdtd_base import *

from gconst_fdtd import *
import fdtd3D_Efaced_dielectric_core

from scipy import zeros, ones, exp

class fdtd3D_Efaced_dielectric:
	def __init__(s, dx, Nx, Ny, Nz):
		s.Nx, s.Ny, s.Nz = Nx, Ny, Nz
		s.dx	=	dx
		s.dt	=	dx/(2*lightV)

		s.ex	=	zeros((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		s.ey	=	zeros((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		s.ez	=	zeros((Nx+2,Ny+2,Nz+2),'f',savespace=1)

		s.hx	=	zeros((Nx+1,Ny+1,Nz+1),'f',savespace=1)
		s.hy	=	zeros((Nx+1,Ny+1,Nz+1),'f',savespace=1)
		s.hz	=	zeros((Nx+1,Ny+1,Nz+1),'f',savespace=1)

		s.epsr_x	=	ones((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		s.epsr_y	=	ones((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		s.epsr_z	=	ones((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		
		s.ceb_x	=	zeros((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		s.ceb_y	=	zeros((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		s.ceb_z	=	zeros((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		
		s.epsr	=	ones((Nx+2,Ny+2,Nz+2),'f',savespace=1)
		

	def set_geometry(s):
		s.epsr_x[1:,:,:]	=	0.5*(s.epsr[1:,:,:] + s.epsr[:-1,:,:])
		s.epsr_y[:,1:,:]	=	0.5*(s.epsr[:,1:,:] + s.epsr[:,:-1,:])
		s.epsr_z[:,:,1:]	=	0.5*(s.epsr[:,:,1:] + s.epsr[:,:,:-1])
		
		"coefficients"
		s.ceb_x		=	1./(2*s.epsr_x)
		s.ceb_y		=	1./(2*s.epsr_y)
		s.ceb_z		=	1./(2*s.epsr_z)
		s.chb		=	0.5
		'''
		# MKS unit
		s.ceb_x		=	1./(2*ceb_x)*(1./imp0)
		s.ceb_y		=	1./(2*ceb_y)*(1./imp0)
		s.ceb_z		=	1./(2*ceb_z)*(1./imp0)
		s.chb		=	0.5*imp0
		'''

		del s.epsr, s.epsr_x, s.epsr_y, s.epsr_z
		
	"======================================================================"
	" FDTD 3D main region with dielectric "
	"======================================================================"
	def updateE(s):
		fdtd3D_Efaced_dielectric_core.updateE(s.Nx,s.Ny,s.Nz,\
									s.ex,s.ey,s.ez,s.hx,s.hy,s.hz,\
									s.ceb_x,s.ceb_y,s.ceb_z)

	def updateH(s):
		fdtd3D_Efaced_dielectric_core.updateH(s.Nx,s.Ny,s.Nz,\
									s.ex,s.ey,s.ez,s.hx,s.hy,s.hz,\
									s.chb)



if __name__ == '__main__':
	from time import *
	from pylab import *
	from scipy import sqrt, sin
	
	S = fdtd3D_Efaced_dielectric(10e-9,45, 45, 45)
	S.set_geometry()
	
	figure(figsize=(12,5))
	show(0)
	
	t0 = time()
	cap_t = 100  # capture_time
	cap_pt = S.Nx/2 # capture_point
	
	for tstep in xrange(1000000):
		S.updateE()
		
		S.ey[25,1:,20] += sin(0.5*tstep)
		
		S.updateH()
		
		if tstep/cap_t*cap_t == tstep:
			t1 = time()
			elapse_time = localtime(t1-t0-60*60*9)
			str_time = strftime('[%j]%H:%M:%S',elapse_time)
			print '%s   tstep = %d' % (str_time,tstep)

			intensity = (S.ex[cap_pt,:,:]**2 + S.ey[cap_pt,:,:]**2 + S.ez[cap_pt,:,:]**2)
			clf()
			imshow(rot90(intensity), cmap=cm.hot, vmin=0, vmax=0.1, interpolation='bilinear')
			colorbar()
			draw()
