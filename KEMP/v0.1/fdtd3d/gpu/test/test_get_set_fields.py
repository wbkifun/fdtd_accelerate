#!/usr/bin/env python

import sys
sys.path.append('/home/kifang')

import numpy as np
import pyopencl as cl
from kemp.fdtd3d import common, common_gpu
from kemp.fdtd3d.gpu import Fields, GetFields, SetFields

randint = np.random.randint


class TestFields(object):
	def __init__(s, nx, ny, nz):
		s.nx = nx
		s.ny = ny
		s.nz = nz

		s.iteration = 5
		s.ns = [s.nx, s.ny, s.nz]
		s.strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']


	def verify(s, pt0, pt1):
		pass


	def set_iteration(s, iteration):
		s.iteration = iteration


	def get_slidx(s, pt0, pt1):
		slidx = [0, 0, 0]
		for i, p0, p1 in zip([0,1,2], pt0, pt1):
			if p0 == p1:
				slidx[i] = p0
			else:
				slidx[i] = slice(p0, p1+1)

		return tuple(slidx)


	def test_point(s):
		print('\n-- test point --')
		for itr in xrange(s.iteration):
			pt0, pt1 = [0, 0, 0], [0, 0, 0]
			for i, n in enumerate(s.ns):
				pt0[i] = pt1[i] = randint(0, n)

			s.verify(pt0, pt1)


	def test_line(s):
		print('\n-- test line --')
		pt0, pt1 = [0, 0, 0], [0, 0, 0]

		for i, j, k in [(0,1,2), (2,0,1), (1,2,0)]:
			pt0[i] = pt1[i] = randint(0, s.ns[i])
			pt0[j] = pt1[j] = randint(0, s.ns[j])
			pt0[k], pt1[k] = 0, s.ns[k]-1
			s.verify(pt0, pt1)

			for itr in xrange(s.iteration):
				pt0[i] = pt1[i] = randint(0, s.ns[i])
				pt0[j] = pt1[j] = randint(0, s.ns[j])
				pt0[k], pt1[k] = np.sort( randint(0, s.ns[k], 2) )
				s.verify(pt0, pt1)


	def test_plane(s):
		print('\n-- test plane --')
		pt0, pt1 = [0, 0, 0], [0, 0, 0]
		pt0, pt1 = [0, 0, 0], [0, 0, 0]

		for i, j, k in [(0,1,2), (2,0,1), (1,2,0)]:
			pt0[i] = pt1[i] = randint(0, s.ns[i])
			pt0[j], pt1[j] = 0, s.ns[j]-1
			pt0[k], pt1[k] = 0, s.ns[k]-1
			s.verify(pt0, pt1)

			for itr in xrange(s.iteration):
				pt0[i] = pt1[i] = randint(0, s.ns[i])
				pt0[j], pt1[j] = np.sort( randint(0, s.ns[j], 2) )
				pt0[k], pt1[k] = np.sort( randint(0, s.ns[k], 2) )
				s.verify(pt0, pt1)


	def test_volume(s):
		print('\n-- test volume --')
		pt0, pt1 = [0, 0, 0], [0, 0, 0]
		pt0, pt1 = [0, 0, 0], [0, 0, 0]

		pt0[0], pt1[0] = 0, s.nx-1
		pt0[1], pt1[1] = 0, s.ny-1
		pt0[2], pt1[2] = 0, s.nz-1
		s.verify(pt0, pt1)

		for itr in xrange(s.iteration):
			pt0[0], pt1[0] = np.sort( randint(0, s.nx, 2) )
			pt0[1], pt1[1] = np.sort( randint(0, s.ny, 2) )
			pt0[2], pt1[2] = np.sort( randint(0, s.nz, 2) )
			s.verify(pt0, pt1)


	def test(s):
		print('iteration = %d' % s.iteration)
		s.test_point()
		s.test_line()
		s.test_plane()
		s.test_volume()



class TestGetFields(TestFields):
	def __init__(s, fdtd, nx, ny, nz):
		super(TestGetFields, s).__init__(nx, ny, nz)
		s.fdtd = fdtd

		s.fhosts = {}
		for strf in s.strf_list:
			s.fhosts[strf] = np.random.rand(nx, ny, nz).astype(s.fdtd.dtype)
			cl.enqueue_write_buffer(s.fdtd.queue, s.fdtd.get_buffer(strf), s.fhosts[strf])


	def verify(s, pt0, pt1):
		print('pt0 = %s, pt1 = %s' % (pt0, pt1))
		slidx = s.get_slidx(pt0, pt1)

		for strf in s.strf_list:
			fget = GetFields(s.fdtd, strf, pt0, pt1)
			fget.get_event().wait()
			original = s.fhosts[strf][slidx]
			copy = fget.get_fields(strf)
			#print original, copy
			assert np.linalg.norm(original - copy) == 0


	def test_boundary(s):
		print('\n-- test boundary (two fields) --')

		print('E fields')
		str_fs_dict = {'x':['ey','ez'], 'y':['ex','ez'], 'z':['ex','ey']}
		pt0 = (0, 0, 0)
		pt1_dict = {'x':(0, s.ny-1, s.nz-1), 'y':(s.nx-1, 0, s.nz-1), 'z':(s.nx-1, s.ny-1, 0)}

		for axis in str_fs_dict.keys():
			print('direction : %s' % axis)
			str_fs = str_fs_dict[axis]
			pt1 = pt1_dict[axis]
			slidx = s.get_slidx(pt0, pt1)
			fget = GetFields(s.fdtd, str_fs, pt0, pt1)
			fget.get_event().wait()

			for strf in str_fs:
				original = s.fhosts[strf][slidx]
				copy = fget.get_fields(strf)
				assert np.linalg.norm(original - copy) == 0


		print('H fields')
		str_fs_dict = {'x':['hy','hz'], 'y':['hx','hz'], 'z':['hx','hy']}
		pt0_dict = {'x':(s.nx-1, 0, 0), 'y':(0, s.ny-1, 0), 'z':(0, 0, s.nz-1)}
		pt1 = (s.nx-1, s.ny-1, s.nz-1)

		for axis in str_fs_dict.keys():
			print('direction : %s' % axis)
			str_fs = str_fs_dict[axis]
			pt0 = pt0_dict[axis]
			slidx = s.get_slidx(pt0, pt1)
			fget = GetFields(s.fdtd, str_fs, pt0, pt1)
			fget.get_event().wait()

			for strf in str_fs:
				original = s.fhosts[strf][slidx]
				copy = fget.get_fields(strf)
				assert np.linalg.norm(original - copy) == 0



class TestSetFields(TestFields):
	def __init__(s, fdtd, nx, ny, nz):
		super(TestSetFields, s).__init__(nx, ny, nz)
		s.fdtd = fdtd

		for strf in s.strf_list:
			randarr = np.random.rand(nx, ny, nz).astype(s.fdtd.dtype)
			cl.enqueue_write_buffer(s.fdtd.queue, s.fdtd.get_buffer(strf), randarr)


	def verify(s, pt0, pt1):
		print('pt0 = %s, pt1 = %s' % (pt0, pt1))
		slidx = s.get_slidx(pt0, pt1)

		for strf in s.strf_list:
			# non-spatial
			fset = SetFields(s.fdtd, strf, pt0, pt1)
			value = np.random.rand()
			fset.set_fields(value)

			fget = GetFields(s.fdtd, strf, pt0, pt1)
			fget.get_event().wait()
			copy = fget.get_fields(strf)

			assert np.linalg.norm(value - copy) == 0

		if pt0 != pt1:
			for strf in s.strf_list:
				# spatial
				fset = SetFields(s.fdtd, strf, pt0, pt1, np.ndarray)
				values = np.random.rand(*fset.rplist['shape']).astype(s.fdtd.dtype)
				fset.set_fields(values)

				fget = GetFields(s.fdtd, strf, pt0, pt1)
				fget.get_event().wait()
				copy = fget.get_fields(strf)

				assert np.linalg.norm(values - copy) == 0


	def test_boundary(s):
		print('\n-- test boundary (two fields) --')
		shape_dict = {'x':(s.ny*2, s.nz), 'y':(s.nx*2, s.nz), 'z':(s.nx*2, s.ny)}

		print('E fields')
		str_fs_dict = {'x':['ey','ez'], 'y':['ex','ez'], 'z':['ex','ey']}
		pt0_dict = {'x':(s.nx-1, 0, 0), 'y':(0, s.ny-1, 0), 'z':(0, 0, s.nz-1)}
		pt1 = (s.nx-1, s.ny-1, s.nz-1)

		for axis in str_fs_dict.keys():
			print('direction : %s' % axis)
			str_fs = str_fs_dict[axis]
			pt0 = pt0_dict[axis]
			slidx = s.get_slidx(pt0, pt1)
			fset = SetFields(s.fdtd, str_fs, pt0, pt1, np.ndarray)
			values = np.random.rand(*shape_dict[axis]).astype(s.fdtd.dtype)
			fset.set_fields(values)

			fget = GetFields(s.fdtd, str_fs, pt0, pt1)
			fget.get_event().wait()
			copy = fget.get_fields()

			assert np.linalg.norm(values - copy) == 0


		print('H fields')
		str_fs_dict = {'x':['hy','hz'], 'y':['hx','hz'], 'z':['hx','hy']}
		pt0 = (0, 0, 0)
		pt1_dict = {'x':(0, s.ny-1, s.nz-1), 'y':(s.nx-1, 0, s.nz-1), 'z':(s.nx-1, s.ny-1, 0)}

		for axis in str_fs_dict.keys():
			print('direction : %s' % axis)
			str_fs = str_fs_dict[axis]
			pt1 = pt1_dict[axis]
			slidx = s.get_slidx(pt0, pt1)
			fset = SetFields(s.fdtd, str_fs, pt0, pt1, np.ndarray)
			values = np.random.rand(*shape_dict[axis]).astype(s.fdtd.dtype)
			fset.set_fields(values)

			fget = GetFields(s.fdtd, str_fs, pt0, pt1)
			fget.get_event().wait()
			copy = fget.get_fields()

			assert np.linalg.norm(values - copy) == 0




if __name__ == '__main__':
	nx, ny, nz = 100, 110, 128
	gpu_id = 0

	gpu_devices = common_gpu.get_gpu_devices()
	context = cl.Context(gpu_devices)
	device = gpu_devices[gpu_id]
	fdtd = Fields(context, device, nx, ny, nz, coeff_use='')

	print('-'*47 + '\nTest GetFields')
	testget = TestGetFields(fdtd, nx, ny, nz)
	testget.set_iteration(3)
	testget.test()
	testget.test_boundary()

	print('-'*47 + '\nTest SetFields')
	testset = TestSetFields(fdtd, nx, ny, nz)
	testset.set_iteration(3)
	testset.test()
	testset.test_boundary()
