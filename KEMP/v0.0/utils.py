#!/usr/bin/env python

import pyopencl as cl
import numpy as np
import sys


def get_gpu_devices(print_verbose=True):
	platforms = cl.get_platforms()
	for platform in platforms:
		if print_verbose:
			print('Platform : %s (%s)' % (platform.get_info(cl.platform_info.NAME), platform.get_info(cl.platform_info.VERSION)))

	if len(platforms) > 1:
		print('Error: %d platforms are found. We are preparing for multi platforms.')
		sys.exit()

	platform = platforms[0]
	gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)

	return gpu_devices


def create_context_queues(devices):
	queues = []
	context = cl.Context(devices)
	for device in devices:
		queues.append( cl.CommandQueue(context, device) )
	
	return context, queues


def print_gpu_info(devices):
	gpu_groups = {}
	for device in devices:
		name = device.get_info(cl.device_info.NAME)
		if not gpu_groups.has_key(name):
			gpu_groups[name] = {'count':1}
			gpu_groups[name]['compute units'] = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
			gpu_groups[name]['global mem size'] = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
			gpu_groups[name]['local mem size'] = device.get_info(cl.device_info.LOCAL_MEM_SIZE)
			gpu_groups[name]['constant mem size'] = device.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE)
		else:
			gpu_groups[name]['count'] += 1

	for name, props in gpu_groups.items():
		print('Device : %d GPU' % props['count'])
		print('  name: %s' % name)
		print('  compute units: %d' % props['compute units'])
		print('  global mem size: %1.2f %s' % get_nbytes_unit(props['global mem size']))
		print('  local mem size: %1.2f %s' % get_nbytes_unit(props['local mem size']))
		print('  constant mem size: %1.2f %s' % get_nbytes_unit(props['constant mem size']))
		print('')


def print_cpu_info():
	for line in open('/proc/cpuinfo'):
		if 'model name' in line:
			cpu_name0 = line[line.find(':')+1:-1]
			break
	cpu_name = ''
	for s in cpu_name0.split():
		cpu_name += (s + ' ')

	for line in open('/proc/meminfo'):
		if 'MemTotal' in line:
			mem_nbytes = int(line[line.find(':')+1:line.rfind('kB')]) * 1024
			break
	print('Host Device :')
	print('  name: %s' % cpu_name)
	print('  mem size: %1.2f %s' % get_nbytes_unit(mem_nbytes))
	print('')


def get_optimal_global_work_size(device):
	warp_size = 32
	max_resident_warp_dict = {
			'1.0':24, '1.1':24,
			'1.2':32, '1.3':32,
			'2.0':48}
	compute_capability = \
			str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV)) \
			+ '.' + str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV))
	max_resident_warp = max_resident_warp_dict[compute_capability]
	max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)

	return max_compute_units * max_resident_warp * warp_size


def get_nbytes_unit(nbytes):
	if nbytes >= 1024**3: 
		value = float(nbytes)/(1024**3)
		unit_str = 'GiB'
	elif nbytes >= 1024**2: 
		value = float(nbytes)/(1024**2)
		unit_str = 'MiB'
	elif nbytes >= 1024: 
		value = float(nbytes)/1024
		unit_str = 'KiB'
	else:
		value = nbytes
		unit_str = 'Bytes'

	return value, unit_str


def print_nbytes(head_str, nx, ny, nz, num_array):
	print('%s (%d, %d, %d)' %(head_str, nx, ny, nz)),
	print('%1.2f %s' % get_nbytes_unit(nx * ny * nz * np.nbytes['float32'] * num_array))


def get_ksrc(path, nx, ny, nz, ls):
	return open(path).read().replace( 'NXYZ',str(nx * ny * nz)).\
			replace('NYZ',str(ny * nz)).\
			replace('NXY',str(nx * ny)).\
			replace('NX',str(nx)).\
			replace('NY',str(ny)).\
			replace('NZ',str(nz)).\
			replace('DX',str(ls))



if __name__ == '__main__':
	gpu_devices = get_gpu_devices()
	print_gpu_info(gpu_devices)
	print_cpu_info()
	context, queues = create_context_queues([gpu_devices[0]])
	print('Optimal Gs = %d' % get_optimal_global_work_size(gpu_devices[0]))
	print_nbytes('sample', 240, 256, 256, 9)


