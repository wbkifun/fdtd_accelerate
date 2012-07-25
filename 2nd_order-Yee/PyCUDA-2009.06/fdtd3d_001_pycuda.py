#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
import scipy as sc
from time import *
from pylab import *

light_velocity = 2.99792458e8;	# m s- 

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

kernel_common = cuda.SourceModule( file('common.cu','r').read() )
kernel_dielectric = cuda.SourceModule( file('dielectric.cu','r').read() )
kernel_source = cuda.SourceModule( file('source.cu','r').read() )
#kernel_cpml = cuda.SourceModule( file('cpml.cu','r').read() )


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


def set_geometry( CEx, CEy, CEz ):
	CEx[1:,1:-1,1:-1] = 0.5
	CEy[1:-1,1:,1:-1] = 0.5
	CEz[1:-1,1:-1,1:] = 0.5


def initMainArrays( Ntot, devFx, devFy, devFz, initArray ):
	TPB = 512
	if ( Ntot%TPB == 0 ):
		BPG = Ntot/TPB
	else:
		BPG = Ntot/TPB + 1
	print 'init main arrays: Ntot=%d, TPB=%d, BPG=%d' % (Ntot, TPB, BPG)

	initArray( sc.int32(Ntot), devFx, block=(TPB,1,1), grid=(BPG,1) )
	initArray( sc.int32(Ntot), devFy, block=(TPB,1,1), grid=(BPG,1) )
	initArray( sc.int32(Ntot), devFz, block=(TPB,1,1), grid=(BPG,1) )




if __name__ == '__main__':
	#for i in xrange( Ndev ):
	#	print_device_attributes( i, devices[i] )
	
	Nx, Ny, Nz = 250, 250, 320
	TMAX = 100000

	S = 0.5
	dx = 10e-9
	dt = S*dx/light_velocity
	
	print 'N(%d,%d,%d), TMAX=%d' % ( Nx, Ny, Nz, TMAX )

	# Allocate host memory
	CEx = sc.zeros( (Nx+1, Ny, Nz), 'f' )
	CEy = sc.zeros( (Nx+1, Ny, Nz), 'f' )
	CEz = sc.zeros( (Nx+1, Ny, Nz), 'f' )

	Ez = sc.zeros( (Nx+2, Ny, Nz), 'f' )

	# Geometry
	set_geometry( CEx, CEy, CEz )

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
	Ns_main = ( 2*(TPB+1)+TPB )*4
	print 'TPB=%d, BPG=%d, Ns_main=%d' % (TPB, BPG, Ns_main)

	# Allocate host memory
	Ntot_devF = ( Nx+2 )*Ny*Nz
	Ntot_devC = ( Nx+1 )*Ny*Nz
	size_devF = Ntot_devF*4
	size_devC = Ntot_devC*4

	devEx = cuda.mem_alloc(size_devF) 
	devEy = cuda.mem_alloc(size_devF) 
	devEz = cuda.mem_alloc(size_devF) 
	devHx = cuda.mem_alloc(size_devF) 
	devHy = cuda.mem_alloc(size_devF) 
	devHz = cuda.mem_alloc(size_devF) 
	devCEx = cuda.mem_alloc(size_devC) 
	devCEy = cuda.mem_alloc(size_devC) 
	devCEz = cuda.mem_alloc(size_devC) 
	
	# Copy arrays from host to device
	cuda.memcpy_htod( devCEx, CEx )
	cuda.memcpy_htod( devCEy, CEy )
	cuda.memcpy_htod( devCEz, CEz )

	# Get kernels
	initArray = kernel_common.get_function("initArray")
	updateE = kernel_dielectric.get_function("updateE")
	updateH = kernel_dielectric.get_function("updateH")
	updateSrc = kernel_source.get_function("updateSrc")
	
	# Initialize the device arrays
	initMainArrays( Ntot_devF, devEx, devEy, devEz, initArray ) 
	initMainArrays( Ntot_devF, devHx, devHy, devHz, initArray ) 

	# Prepare to call the kernels
	kNx, kNy, kNz = sc.int32(Nx), sc.int32(Ny), sc.int32(Nz)
	updateE.prepare( "iiPPPPPPPPP", block=(TPB,1,1), shared=Ns_main )
	updateSrc.prepare( "iiiiP", block=(Nz,1,1) )
	updateH.prepare( "iiPPPPPP", block=(TPB,1,1), shared=Ns_main )
	'''
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
	'''

	# time loop
	t0 = time()
	for tstep in xrange( 1, 100001 ):
		updateE.prepared_call( (BPG,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz )

		updateSrc.prepared_call( (1,1), kNx, kNy, kNz, sc.int32(tstep), devEz );

		updateH.prepared_call( (BPG,1), kNy, kNz, devEx, devEy, devEz, devHx, devHy, devHz )

		if tstep/1000*1000 == tstep:
			t1 = time()
			elapse_time = localtime(t1-t0-60*60*9)
			str_time = strftime('[%j]%H:%M:%S', elapse_time)
			print '%s    tstep = %d' % (str_time, tstep)

			'''
			cuda.memcpy_dtoh( Ez, devEz )
			imsh.set_array( Ez[:,:,Nz/2] )
			png_str = './gpu_png/Ez-%.6d.png' % tstep
			savefig(png_str) 
			'''
