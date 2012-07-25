'''
 Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
 		  Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
'''

from kufdtd.common import *
from kufdtd.dim3.gpu.base import *


class Matter( GpuSpace ):
	def __init__( s, Nx, Ny, Nz, dx ):
		GpuSpace.__init__( s, Nx, Ny, Nz, dx )

		s.size = s.Nx*s.Ny*s.Nz
		s.size1 = ( s.Nx+1 )*s.Ny*s.Nz
		s.size2 = ( s.Nx+2 )*s.Ny*s.Nz

		s.bytes = s.size*s.bytes_f
		s.bytes1 = s.size1*s.bytes_f
		s.bytes2 = s.size2*s.bytes_f

		s.verify_16xNz()


	def allocate_main_in_dev( s ):
		s.devEx = cuda.mem_alloc( s.bytes2 )
		s.devEy = cuda.mem_alloc( s.bytes2 )
		s.devEz = cuda.mem_alloc( s.bytes2 )
		s.devHx = cuda.mem_alloc( s.bytes2 )
		s.devHy = cuda.mem_alloc( s.bytes2 )
		s.devHz = cuda.mem_alloc( s.bytes2 )


	def free_main_in_dev( s ):
		s.devEx.free()
		s.devEy.free()
		s.devEz.free()
		s.devHx.free()
		s.devHy.free()
		s.devHz.free()


	def initmem_main_in_dev( s ):
		initmem = s.get_kernel_initmem()

		tpb = 512
		Db = (tpb,1,1)

		nk, sub_bpgs, idx0 = s.calc_sub_bpgs( s.size2, tpb )
		Ntot = sc.int32( s.size2 )

		for i in xrange( nk ):
			i0 = sc.int32( idx0[i] )

			initmem( Ntot, i0, s.devEx, block=Db, grid=(sub_bpgs[i],1) )
			initmem( Ntot, i0, s.devEy, block=Db, grid=(sub_bpgs[i],1) )
			initmem( Ntot, i0, s.devEz, block=Db, grid=(sub_bpgs[i],1) )
						        	  
			initmem( Ntot, i0, s.devHx, block=Db, grid=(sub_bpgs[i],1) )
			initmem( Ntot, i0, s.devHy, block=Db, grid=(sub_bpgs[i],1) )
			initmem( Ntot, i0, s.devHz, block=Db, grid=(sub_bpgs[i],1) )
		                      

	def allocate_coeff_in_dev( s ): 
		s.devCEx = cuda.mem_alloc( s.bytes1 ) 
		s.devCEy = cuda.mem_alloc( s.bytes1 )
		s.devCEz = cuda.mem_alloc( s.bytes1 )


	def free_coeff_in_dev( s ):
		s.devCEx.free()
		s.devCEy.free()
		s.devCEz.free()


class Dielectric( Matter ):
	def __init__( s, Nx, Ny, Nz, dx ):
		Matter.__init__( s, Nx, Ny, Nz, dx )

		s.set_kernel_parameters()


	def set_kernel_parameters( s ):
		Ntot = s.size
		s.tpb, s.occupancy = s.select_tpb( Ntot, s.Ny*s.Nz )
		s.nk, s.sub_bpgs, s.idx0 = s.calc_sub_bpgs( Ntot, s.tpb )

		s.ns = ( 2*(s.tpb+1)+s.tpb )*s.bytes_f


	def print_kernel_parameters( s ):
		print 'main: tpb=%d, sub_bpg(%d)=%s' % (s.tpb, s.nk, s.sub_bpgs)
		print '      occupancy=%1.2f, ns=%d' % (s.occupancy, s.ns)


	def print_memory_usage( s ):
		mbytes = 1024**2
		eh = 6*s.bytes2/mbytes
		ce = 3*s.bytes1/mbytes

		print 'memory usage: %d Mbytes (E,H=%d, CE=%d)' % ( eh+ce, eh, ce ) 


	def allocate_coeff( s ):
		shape = (s.Nx+1, s.Ny, s.Nz)
		s.CEx = sc.zeros( shape, 'f' )
		s.CEy = sc.zeros( shape, 'f' )
		s.CEz = sc.zeros( shape, 'f' )


	def free_coeff( s ):
		del s.CEx
		del s.CEy
		del s.CEz 


	def set_coeff( s ):
		s.CEx[1:1,1:-1,1:-1] = 0.5
		s.CEy[1:-2,1:,1:-1] = 0.5
		s.CEz[1:-2,1:-1,1:] = 0.5


	def memcpy_htod_coeff( s ):
		cuda.memcpy_htod( s.devCEx, s.CEx )
		cuda.memcpy_htod( s.devCEy, s.CEy )
		cuda.memcpy_htod( s.devCEz, s.CEz )


	def prepare_kernels( s ):
		fpath = '%s/core/dielectric.cu' % base_dir
		mod = cuda.SourceModule( file( fpath,'r' ).read() )
		s.update_e = mod.get_function("update_e")
		s.update_h = mod.get_function("update_h")

		Db = ( s.tpb, 1, 1 )
		s.update_e.prepare( "iiiPPPPPPPPP", block=Db, shared=s.ns )
		s.update_h.prepare( "iiiPPPPPP", block=Db, shared=s.ns )

		s.kNyz = sc.int32( s.Ny*s.Nz )


	def updateE( s ):
		for i in xrange( s.nk ):
			s.update_e.prepared_call( \
					( s.sub_bpgs[i], 1 ), \
					s.idx0[i], s.kNz, s.kNyz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz, 
					s.devCEx, s.devCEy, s.devCEz )
	

	def updateH( s ):
		for i in xrange( s.nk ):
			s.update_h.prepared_call( \
					( s.sub_bpgs[i], 1 ), \
					s.idx0[i], s.kNz, s.kNyz, \
					s.devEx, s.devEy, s.devEz, \
					s.devHx, s.devHy, s.devHz )
