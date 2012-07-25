import numpy as np


class FDTD3DGPU:
	def __init__(s, nx, ny, nz, gpu_num, cuda_drv):
		s.nx, s.ny, s.nz = nx, ny, nz
		s.Dx, s.Dy = 32, 16

		if s.nz%s.Dx != 0:
			print "Error: nz is not multiple of %d" % (s.Dx)
			sys.exit()
		if (s.nx*s.ny)%s.Dy != 0:
			print "Error: nx*ny is not multiple of %d" % (s.Dy)
			sys.exit()

		s.cuda = cuda_drv
		s.dev = s.cuda.Device(gpu_num)
		s.ctx = s.dev.make_context()
		s.MAX_BLOCK = s.dev.get_attribute(s.cuda.device_attribute.MAX_GRID_DIM_X)


	def finalize(s):
		s.ctx.pop()


	def alloc_eh_fields(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.ex_gpu = s.cuda.to_device(f)
		s.ey_gpu = s.cuda.to_device(f)
		s.ez_gpu = s.cuda.to_device(f)
		s.hx_gpu = s.cuda.to_device(f)
		s.hy_gpu = s.cuda.to_device(f)
		s.hz_gpu = s.cuda.to_device(f)
		s.eh_fields = [s.ex_gpu, s.ey_gpu, s.ez_gpu, s.hx_gpu, s.hy_gpu, s.hz_gpu]


	def alloc_coeff_arrays(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.cex = np.ones_like(f)*0.5
		s.cex[:,-1,:] = 0
		s.cex[:,:,-1] = 0
		s.cey = np.ones_like(f)*0.5
		s.cey[:,:,-1] = 0
		s.cey[-1,:,:] = 0
		s.cez = np.ones_like(f)*0.5
		s.cez[-1,:,:] = 0
		s.cez[:,-1,:] = 0

		descr = s.cuda.ArrayDescriptor3D()
		descr.width = s.nz
		descr.height = s.ny
		descr.depth = s.nx
		descr.format = s.cuda.dtype_to_array_format(f.dtype)
		descr.num_channels = 1
		descr.flags = 0
		s.tcex_gpu = s.cuda.Array(descr)
		s.tcey_gpu = s.cuda.Array(descr)
		s.tcez_gpu = s.cuda.Array(descr)

		mcpy = s.cuda.Memcpy3D()
		mcpy.width_in_bytes = mcpy.src_pitch = f.strides[1]
		mcpy.src_height = mcpy.height = s.ny
		mcpy.depth = s.nx
		mcpy.set_src_host( s.cex )
		mcpy.set_dst_array( s.tcex_gpu )
		mcpy()
		mcpy.set_src_host( s.cey )
		mcpy.set_dst_array( s.tcey_gpu )
		mcpy()
		mcpy.set_src_host( s.cez )
		mcpy.set_dst_array( s.tcez_gpu )
		mcpy()


	def alloc_exchange_boundaries(s):
		s.ey_tmp = s.cuda.pagelocked_zeros((s.ny,s.nz),'f')
		s.ez_tmp = s.cuda.pagelocked_zeros_like(s.ey_tmp)
		s.hy_tmp = s.cuda.pagelocked_zeros_like(s.ey_tmp)
		s.hz_tmp = s.cuda.pagelocked_zeros_like(s.ey_tmp)


	def prepare_functions(s):
		from pycuda.compiler import SourceModule
		kernels = ''.join( open("dielectric.cu",'r').readlines() )
		mod = SourceModule( kernels.replace('Dx',str(s.Dx)).replace('Dy',str(s.Dy)).replace('nyz',str(s.ny*s.nz)).replace('nx',str(s.nx)).replace('ny',str(s.ny)).replace('nz',str(s.nz)) )
		s.updateH = mod.get_function("update_h")
		s.updateE = mod.get_function("update_e")
		s.updateE_src = mod.get_function("update_src")

		tcex = mod.get_texref("tcex")
		tcey = mod.get_texref("tcey")
		tcez = mod.get_texref("tcez")
		tcex.set_array(s.tcex_gpu)
		tcey.set_array(s.tcey_gpu)
		tcez.set_array(s.tcez_gpu)

		Bx, By = s.nz/s.Dx, s.nx*s.ny/s.Dy	# number of block
		s.MaxBy = s.MAX_BLOCK/Bx
		s.bpg_list = [(Bx,s.MaxBy) for i in range(By/s.MaxBy)]
		if By%s.MaxBy != 0: s.bpg_list.append( (Bx,By%s.MaxBy) )

		s.updateH.prepare("iPPPPPP", block=(s.Dx,s.Dy,1))
		s.updateE.prepare("iPPPPPP", block=(s.Dx,s.Dy,1), texrefs=[tcex,tcey,tcez])
		s.updateE_src.prepare("fP", block=(s.nz,1,1))


	def update_h(s):
		for i, bpg in enumerate(s.bpg_list): s.updateH.prepared_call(bpg, np.int32(i*s.MaxBy), *s.eh_fields)


	def update_e(s):
		for i, bpg in enumerate(s.bpg_list): s.updateE.prepared_call(bpg, np.int32(i*s.MaxBy), *s.eh_fields)


	def update_src(s, tn):
		s.updateE_src.prepared_call((1,1), np.float32(tn), s.ez_gpu)


	def mpi_exchange_boundary_h(s, mpi_direction, comm):
		if 'f' in mpi_direction:
			comm.Recv(s.hy_tmp, comm.rank-1, 0)
			comm.Recv(s.hz_tmp, comm.rank-1, 1)
			s.cuda.memcpy_htod(int(s.hy_gpu), s.hy_tmp) 
			s.cuda.memcpy_htod(int(s.hz_gpu), s.hz_tmp) 
		if 'b' in mpi_direction:
			s.cuda.memcpy_dtoh(s.hy_tmp, int(s.hy_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32']) 
			s.cuda.memcpy_dtoh(s.hz_tmp, int(s.hz_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32']) 
			comm.Send(s.hy_tmp, comm.rank+1, 0)
			comm.Send(s.hz_tmp, comm.rank+1, 1)


	def mpi_exchange_boundary_e(s, mpi_direction, comm):
		if 'f' in mpi_direction:
			s.cuda.memcpy_dtoh(s.ey_tmp, int(s.ey_gpu)) 
			s.cuda.memcpy_dtoh(s.ez_tmp, int(s.ez_gpu)) 
			comm.Send(s.ey_tmp, comm.rank-1, 2)
			comm.Send(s.ez_tmp, comm.rank-1, 3)
		if 'b' in mpi_direction:
			comm.Recv(s.ey_tmp, comm.rank+1, 2)
			comm.Recv(s.ez_tmp, comm.rank+1, 3)
			s.cuda.memcpy_htod(int(s.ey_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32'], s.ey_tmp) 
			s.cuda.memcpy_htod(int(s.ez_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32'], s.ez_tmp) 



class FDTD3DCPU:
	def __init__(s, nx, ny, nz):
		s.nx, s.ny, s.nz = nx, ny, nz


	def finalize(s):
		pass


	def alloc_eh_fields(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.ex = f.copy()
		s.ey = f.copy()
		s.ez = f.copy()
		s.hx = f.copy()
		s.hy = f.copy()
		s.hz = f.copy()
		s.eh_fields = [s.ex, s.ey, s.ez, s.hx, s.hy, s.hz]


	def alloc_coeff_arrays(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.cex = np.ones_like(f)*0.5
		s.cex[:,-1,:] = 0
		s.cex[:,:,-1] = 0
		s.cey = np.ones_like(f)*0.5
		s.cey[:,:,-1] = 0
		s.cey[-1,:,:] = 0
		s.cez = np.ones_like(f)*0.5
		s.cez[-1,:,:] = 0
		s.cez[:,-1,:] = 0
		s.ce_arrays = [s.cex, s.cey, s.cez]


	def alloc_exchange_boundaries(s):
		s.ey_tmp = np.zeros((s.ny,s.nz),'f')
		s.ez_tmp = np.zeros_like(s.ey_tmp)
		s.hy_tmp = np.zeros_like(s.ey_tmp)
		s.hz_tmp = np.zeros_like(s.ey_tmp)


	def prepare_functions(s):
		import os
		os.system("gcc -O3 -fpic -fopenmp -msse -I/usr/include/python2.6 -I/usr/lib/python2.6/site-packages/numpy/core/include -shared dielectric.c -o dielectric.so")
		#os.system("gcc -O3 -fpic -msse -I/usr/include/python2.6 -I/usr/lib/python2.6/site-packages/numpy/core/include -shared dielectric.c -o dielectric.so")
		import dielectric
		s.updateH = dielectric.update_h
		s.updateE = dielectric.update_e


	def update_h(s):
		s.updateH( *s.eh_fields )


	def update_e(s):
		s.updateE( *(s.eh_fields + s.ce_arrays) )


	def update_src(s, tn):
		s.ez[s.nx/2,s.ny/2,:] += np.sin(0.1*tn)


	def mpi_exchange_boundary_h(s, mpi_direction, comm):
		if 'f' in mpi_direction:
			comm.Recv(s.hy_tmp, comm.rank-1, 0)
			comm.Recv(s.hz_tmp, comm.rank-1, 1)
			s.hy[0,:,:] = s.hy_tmp[:,:] 
			s.hz[0,:,:] = s.hz_tmp[:,:] 
		if 'b' in mpi_direction:
			s.hy_tmp[:,:] = s.hy[-1,:,:] 
			s.hz_tmp[:,:] = s.hz[-1,:,:] 
			comm.Send(s.hy_tmp, comm.rank+1, 0)
			comm.Send(s.hz_tmp, comm.rank+1, 1)


	def mpi_exchange_boundary_e(s, mpi_direction, comm):
		if 'f' in mpi_direction:
			s.ey_tmp[:,:] = s.ey[0,:,:] 
			s.ez_tmp[:,:] = s.ez[0,:,:] 
			comm.Send(s.ey_tmp, comm.rank-1, 2)
			comm.Send(s.ez_tmp, comm.rank-1, 3)
		if 'b' in mpi_direction:
			comm.Recv(s.ey_tmp, comm.rank+1, 2)
			comm.Recv(s.ez_tmp, comm.rank+1, 3)
			s.ey[-1,:,:] = s.ey_tmp[:,:] 
			s.ez[-1,:,:] = s.ez_tmp[:,:] 
