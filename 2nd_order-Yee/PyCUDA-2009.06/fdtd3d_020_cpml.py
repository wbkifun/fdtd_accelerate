#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit

import scipy as sc
from time import *
from pylab import *
import sys

light_velocity = 2.99792458e8	# m s- 
ep0 = 8.85418781762038920e-12	# F m-1 (permittivity at vacuum)
mu0 = 1.25663706143591730e-6	# N A-2 (permeability at vacuum)
imp0 = sc.sqrt( mu0/ep0 )		# (impedance at vacuum)
pi = 3.14159265358979323846


'''
cuda.init()
assert cuda.Device.count() >= 1

Ndev = cuda.Device.count()
print 'Number of devices = %d' % ( Ndev )

devices = []
for i in xrange( Ndev ):
	devices.append( cuda.Device(i) )

dev = cuda.Device(0)
ctx = dev.make_context()
'''


def print_device_attributes( i, dev ):
	atts = dev.get_attributes()

	print 'Device %d: \"%s\"' % ( i, dev.name() )
	print '  Compute Capability: \t\t\t\t%d.%d' % dev.compute_capability()
	print '  Total amount of global memory:\t\t%d bytes' % dev.total_memory()
	sm_count = atts[pycuda._driver.device_attribute.MULTIPROCESSOR_COUNT]
	print '  Number of multiprocessors:\t\t\t%d' % sm_count
	print '  Number of cores:\t\t\t\t%d' % (sm_count*8)
	print '  Total amount of constant memory:\t\t%d bytes' % atts[pycuda._driver.device_attribute.TOTAL_CONSTANT_MEMORY]
	print '  Total amount of shared memory per block:\t%d bytes' % atts[pycuda._driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
	print '  Total number of registers available per block:%d' % atts[pycuda._driver.device_attribute.MAX_REGISTERS_PER_BLOCK]
	print '  Warp size:\t\t\t\t\t%d' % atts[pycuda._driver.device_attribute.WARP_SIZE]
	print '  Maximum number of threads per block:\t\t%d' % atts[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
	print '  Maximum number of blocks per grid:\t\t%d' % atts[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
	print '  Maximum memory pitch:\t\t\t\t%d bytes' % atts[pycuda._driver.device_attribute.MAX_PITCH]
	print '  Texture alignment:\t\t\t\t%d bytes' % atts[pycuda._driver.device_attribute.TEXTURE_ALIGNMENT]
	print '  Clock rate:\t\t\t\t\t%d KHz' % atts[pycuda._driver.device_attribute.CLOCK_RATE]
	if ( atts[pycuda._driver.device_attribute.GPU_OVERLAP] ):
		gpu_overlap = 'yes'
	else:
		gpu_overlap = 'no'
	print '  Concurrent copy and execution:\t\t%s' % gpu_overlap
	print ''


def verify_16xNz( Nz ):
	R = Nz%16
	if ( R == 0 ):
		print 'Nz is a multiple of 16.'
	else:
		print 'Error: Nz is not a multiple of 16.'
		print 'Recommend Nz: %d or %d' % (Nz-R, Nz-R+16)
		sys.exit(0)


def set_geometry( CEx, CEy, CEz ):
	CEx[1:,1:-1,1:-1] = 0.5
	CEy[1:-1,1:,1:-1] = 0.5
	CEz[1:-1,1:-1,1:] = 0.5


def initMainArrays( kNtot, devFx, devFy, devFz, initArray ):
	TPB = 512
	if ( Ntot%TPB == 0 ):
		BPG = Ntot/TPB
	else:
		BPG = Ntot/TPB + 1
	print 'init main arrays: Ntot=%d, TPB=%d, BPG=%d' % (Ntot, TPB, BPG)

	initArray( kNtot, devFx, block=(TPB,1,1), grid=(BPG,1) )
	initArray( kNtot, devFy, block=(TPB,1,1), grid=(BPG,1) )
	initArray( kNtot, devFz, block=(TPB,1,1), grid=(BPG,1) )


def initPsiArrays( kNtot, TPB, BPG, psi1f, psi1b, psi2f, psi2b, initArray ):
	initArray( kNtot, psi1f, block=(TPB,1,1), grid=(BPG,1) )
	initArray( kNtot, psi1b, block=(TPB,1,1), grid=(BPG,1) )
	initArray( kNtot, psi2f, block=(TPB,1,1), grid=(BPG,1) )
	initArray( kNtot, psi2b, block=(TPB,1,1), grid=(BPG,1) )



if __name__ == '__main__':
	#for i in xrange( Ndev ):
	#	print_device_attributes( i, devices[i] )
	
	Nx, Ny, Nz = 250, 250, 320
	TMAX = 100000

	S = 0.5
	dx = 10e-9
	dt = S*dx/light_velocity
	
	print 'N(%d,%d,%d), TMAX=%d' % ( Nx, Ny, Nz, TMAX )


	# Allocate the host memory
	CEx = sc.zeros( (Nx+1, Ny, Nz), 'f' )
	CEy = sc.zeros( (Nx+1, Ny, Nz), 'f' )
	CEz = sc.zeros( (Nx+1, Ny, Nz), 'f' )

	Ez = sc.zeros( (Nx+2, Ny, Nz), 'f' )


	# Geometry
	set_geometry( CEx, CEy, CEz )


	# Parameters for CPML
	Npml = 15
	print "Npml=%d\n" % Npml
	m = 4	# grade_order
	sigma_max = (m+1.)/(15*pi*Npml*dx)
	alpha = 0.05

	sigmaE = sc.zeros( 2*(Npml+1), 'f')
	sigmaH = sc.zeros( 2*(Npml+1), 'f')
	bE = sc.zeros( 2*(Npml+1), 'f')
	bH = sc.zeros( 2*(Npml+1), 'f')
	aE = sc.zeros( 2*(Npml+1), 'f')
	aH = sc.zeros( 2*(Npml+1), 'f')
	for i in xrange(Npml):
		sigmaE[i] = pow( (Npml-0.5-i)/Npml, m )*sigma_max
		sigmaE[i+Npml+1] = pow( (0.5+i)/Npml, m )*sigma_max
		sigmaH[i+1] = pow( float(Npml-i)/Npml, m )*sigma_max
		sigmaH[i+Npml+2] = pow( (1.+i)/Npml, m )*sigma_max

	bE[:] = sc.exp( -(sigmaE[:] + alpha)*dt/ep0 );
	bH[:] = sc.exp( -(sigmaH[:] + alpha)*dt/ep0 );
	aE[:] = sigmaE[:]/(sigmaE[:]+alpha)*(bE[:]-1);
	aH[:] = sigmaH[:]/(sigmaH[:]+alpha)*(bH[:]-1);
	
	del sigmaE, sigmaH


	# Set the GPU kernel parameters
	# TPB: Number of threads per block
	# BPG: Number of thread blocks per grid
	# main 
	Ntot = Nx*Ny*Nz
	TPB = 512
	if ( Ntot%TPB == 0 ):
		BPG = Ntot/TPB
	else:
		BPG = Ntot/TPB + 1
	BPGmain = BPG
	Ns_main = ( 2*(TPB+1)+TPB )*4
	print 'TPB=%d, BPGmain=%d, Ns_main=%d' % (TPB, BPGmain, Ns_main)

	# cpml
	TPBpmlx = TPBpmly = TPBpmlz = TPB

	Ntot = Npml*Ny*Nz
	if ( Ntot%TPB == 0 ):
		BPG = Ntot/TPB
	else:
		BPG = Ntot/TPB + 1
	BPGpmlx = BPG
	Ntotpmlx = TPB*BPG
	print 'BPGpmlx=%d' % (BPGpmlx)

	Ntot = Nx*Npml*Nz
	if ( Ntot%TPB == 0 ):
		BPG = Ntot/TPB
	else:
		BPG = Ntot/TPB + 1
	BPGpmly = BPG
	Ntotpmly = TPB*BPG
	print 'BPGpmly=%d' % (BPGpmly)

	Ntot = Nx*Ny*(Npml+1)
	if ( Ntot%TPB == 0 ):
		BPG = Ntot/TPB
	else:
		BPG = Ntot/TPB + 1
	BPGpmlz = BPG
	Ntotpmlz = TPB*BPG
	Ns_pmlz = ( 2*(TPB+1) )*4
	print 'BPGpmlz=%d, Ns_pmlz=%d' % (BPGpmlz, Ns_pmlz)


	# Allocate the device memory
	Ntot_devF = ( Nx+2 )*Ny*Nz
	Ntot_devC = ( Nx+1 )*Ny*Nz
	size_devF = Ntot_devF*4
	size_devC = Ntot_devC*4

	devEx = cuda.mem_alloc( size_devF ) 
	devEy = cuda.mem_alloc( size_devF ) 
	devEz = cuda.mem_alloc( size_devF ) 
	devHx = cuda.mem_alloc( size_devF ) 
	devHy = cuda.mem_alloc( size_devF ) 
	devHz = cuda.mem_alloc( size_devF ) 
	devCEx = cuda.mem_alloc( size_devC ) 
	devCEy = cuda.mem_alloc( size_devC ) 
	devCEz = cuda.mem_alloc( size_devC ) 
	

	# Allocate the device memory for CPML
	size_psix = Ntotpmlx*4
	size_psiy = Ntotpmly*4
	size_psiz = Ntotpmlz*4

	psixEyf = cuda.mem_alloc( size_psix )
	psixEyb = cuda.mem_alloc( size_psix )
	psixEzf = cuda.mem_alloc( size_psix )
	psixEzb = cuda.mem_alloc( size_psix )
	psixHyf = cuda.mem_alloc( size_psix )
	psixHyb = cuda.mem_alloc( size_psix )
	psixHzf = cuda.mem_alloc( size_psix )
	psixHzb = cuda.mem_alloc( size_psix )

	psiyEzf = cuda.mem_alloc( size_psiy )
	psiyEzb = cuda.mem_alloc( size_psiy )
	psiyExf = cuda.mem_alloc( size_psiy )
	psiyExb = cuda.mem_alloc( size_psiy )
	psiyHzf = cuda.mem_alloc( size_psiy )
	psiyHzb = cuda.mem_alloc( size_psiy )
	psiyHxf = cuda.mem_alloc( size_psiy )
	psiyHxb = cuda.mem_alloc( size_psiy )

	psizExf = cuda.mem_alloc( size_psiz )
	psizExb = cuda.mem_alloc( size_psiz )
	psizEyf = cuda.mem_alloc( size_psiz )
	psizEyb = cuda.mem_alloc( size_psiz )
	psizHxf = cuda.mem_alloc( size_psiz )
	psizHxb = cuda.mem_alloc( size_psiz )
	psizHyf = cuda.mem_alloc( size_psiz )
	psizHyb = cuda.mem_alloc( size_psiz )


	# Copy the arrays from host to device
	cuda.memcpy_htod( devCEx, CEx )
	cuda.memcpy_htod( devCEy, CEy )
	cuda.memcpy_htod( devCEz, CEz )


	# Get the module from the cuda files
	mod_common = cuda.SourceModule( file('common.cu','r').read() )
	mod_dielectric = cuda.SourceModule( file('dielectric.cu','r').read() )
	mod_source = cuda.SourceModule( file('source.cu','r').read() )
	mod_cpml = cuda.SourceModule( file('cpml.cu','r').read().replace('NPMLp2',str(2*(Npml+1))).replace('NPMLp',str(Npml+1)).replace('NPML',str(Npml)  ) )


	# Get the global pointer from the module
	rcmbE = mod_cpml.get_global("rcmbE")
	rcmbH = mod_cpml.get_global("rcmbH")
	rcmaE = mod_cpml.get_global("rcmaE")
	rcmaH = mod_cpml.get_global("rcmaH")

	#print rcmaE
	#print bE

	# Copy the arrays from host to constant memory
	cuda.memcpy_htod( rcmbE[0], bE )
	cuda.memcpy_htod( rcmbH[0], bH )
	cuda.memcpy_htod( rcmaE[0], aE )
	cuda.memcpy_htod( rcmaH[0], aH )

	#tmp = sc.zeros( 2*(Npml+1), 'f')
	#cuda.memcpy_dtoh( tmp, rcmbE[0] )
	#print tmp[:] - bE[:]

	# Get the kernel from the modules
	initArray = mod_common.get_function("initArray")
	updateE = mod_dielectric.get_function("updateE")
	updateH = mod_dielectric.get_function("updateH")
	updateSrc = mod_source.get_function("updateSrc")
	updateCPMLxE = mod_cpml.get_function("updateCPMLxE")
	updateCPMLxH = mod_cpml.get_function("updateCPMLxH")
	updateCPMLyE = mod_cpml.get_function("updateCPMLyE")
	updateCPMLyH = mod_cpml.get_function("updateCPMLyH")
	updateCPMLzE = mod_cpml.get_function("updateCPMLzE")
	updateCPMLzH = mod_cpml.get_function("updateCPMLzH")
	

	# Initialize the device arrays
	kNtot_devF = sc.int32(Ntot_devF)
	initMainArrays( kNtot_devF, devEx, devEy, devEz, initArray ) 
	initMainArrays( kNtot_devF, devHx, devHy, devHz, initArray ) 

	kNtotpmlx = sc.int32(Ntotpmlx)
	kNtotpmly = sc.int32(Ntotpmly)
	kNtotpmlz = sc.int32(Ntotpmlz)
	initPsiArrays( kNtotpmlx, TPBpmlx, BPGpmlx, psixEyf, psixEyb, psixEzf, psixEzb, initArray ) 
	initPsiArrays( kNtotpmlx, TPBpmlx, BPGpmlx, psixHyf, psixHyb, psixHzf, psixHzb, initArray ) 
	initPsiArrays( kNtotpmly, TPBpmly, BPGpmly, psiyEzf, psiyEzb, psiyExf, psiyExb, initArray ) 
	initPsiArrays( kNtotpmly, TPBpmly, BPGpmly, psiyHzf, psiyHzb, psiyHxf, psiyHxb, initArray ) 
	initPsiArrays( kNtotpmlz, TPBpmlz, BPGpmlz, psizExf, psizExb, psizEyf, psizEyb, initArray ) 
	initPsiArrays( kNtotpmlz, TPBpmlz, BPGpmlz, psizHxf, psizHxb, psizHyf, psizHyb, initArray ) 


	# Prepare to call the kernels
	kNx, kNy, kNz = sc.int32(Nx), sc.int32(Ny), sc.int32(Nz)
	updateE.prepare( "iiPPPPPPPPP", block=(TPB,1,1), shared=Ns_main )
	updateH.prepare( "iiPPPPPP", block=(TPB,1,1), shared=Ns_main )

	updateSrc.prepare( "iiiiP", block=(Nz,1,1) )

	updateCPMLxE.prepare( "iiiPPPPPPPPPPPi", block=(TPBpmlx,1,1) )
	updateCPMLyE.prepare( "iiPPPPPPPPPPPi", block=(TPBpmly,1,1) )
	updateCPMLzE.prepare( "iiPPPPPPPPPPPi", block=(TPBpmlz,1,1), shared=Ns_pmlz )
	updateCPMLxH.prepare( "iiiPPPPPPPPi", block=(TPBpmlx,1,1) )
	updateCPMLyH.prepare( "iiPPPPPPPPi", block=(TPBpmly,1,1) )
	updateCPMLzH.prepare( "iiPPPPPPPPi", block=(TPBpmlz,1,1), shared=Ns_pmlz )


	# Graphic
	ion()
	figure()

	Ez[:,:,Nz/2] = 1
	imsh = imshow( Ez[:,:,Nz/2],
					cmap=cm.jet,
					vmin=-0.05, vmax=0.05,
					origin='lower',
					interpolation='bilinear')
	colorbar()

	# time loop
	t0 = time()
	for tstep in xrange( 1, 201 ):
		updateE.prepared_call( (BPGmain,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz )
		
		#updateCPMLxE.prepared_call( (BPGpmlx,1), kNx, kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz, psixEyf, psixEzf, sc.int32(0) )
		'''
		updateCPMLxE.prepared_call( (BPGpmlx,1), kNx, kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz, psixEyb, psixEzb, 1 )
		updateCPMLyE.prepared_call( (BPGpmly,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz, psiyEzf, psiyExf, 0 )
		updateCPMLyE.prepared_call( (BPGpmly,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz, psiyEzb, psiyExb, 1 )
		updateCPMLzE.prepared_call( (BPGpmlz,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz, psizExf, psizEyf, 0 )
		updateCPMLzE.prepared_call( (BPGpmlz,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz, psizExb, psizEyb, 1 )
		'''

		updateSrc.prepared_call( (1,1), kNx, kNy, kNz, sc.int32(tstep), devEz );

		updateH.prepared_call( (BPGmain,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz )

		#updateCPMLxH.prepared_call( (BPGpmlx,1), kNx, kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, psixHyf, psixHzf, sc.int32(0) )
		'''
		updateCPMLxH.prepared_call( (BPGpmlx,1), kNx, kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, psixHyb, psixHzb, 1 )
		updateCPMLyH.prepared_call( (BPGpmly,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, psiyHzf, psiyHxf, 0 )
		updateCPMLyH.prepared_call( (BPGpmly,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, psiyHzb, psiyHxb, 1 )
		updateCPMLzH.prepared_call( (BPGpmlz,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, psizHxf, psizHyf, 0 )
		updateCPMLzH.prepared_call( (BPGpmlz,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, psizHxb, psizHyb, 1 )
		'''

		if tstep/10*10 == tstep:
			t1 = time()
			elapse_time = localtime(t1-t0-60*60*9)
			str_time = strftime('[%j]%H:%M:%S', elapse_time)
			print '%s    tstep = %d' % (str_time, tstep)

			cuda.memcpy_dtoh( Ez, devEz )
			imsh.set_array( Ez[:,:,Nz/2] )
			png_str = './gpu_png/Ez-%.6d.png' % tstep
			savefig(png_str) 
