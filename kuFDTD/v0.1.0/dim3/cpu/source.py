from kufdtd.common import *
from kufdtd.dim3.gpu.base import *


class Source( GpuSpace ):
	def __init__( s, main_space ):
		MS = main_space
		GpuSpace.__init__( s, MS.Nx, MS.Ny, MS.Nz, MS.dx )

		s.set_kernel_parameters()


	def set_kernel_parameters( s ):
		s.tpb = s.Nz
		s.bpg = 1


	def prepare_kernels( s ):
		fpath = '%s/core/source.cu' % base_dir
		mod = cuda.SourceModule( file( fpath,'r' ).read() )
		s.update_src = mod.get_function("update_src")

		Db = ( s.tpb, 1, 1 )
		s.update_src.prepare( "iiiiP", block=Db )


	def updateE( s, tstep, F ):
		s.update_src.prepared_call( (s.bpg,1), s.kNx, s.kNy, s.kNz, sc.int32(tstep), F )
