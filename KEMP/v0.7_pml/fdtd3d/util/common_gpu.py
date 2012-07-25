import pyopencl as cl
import os
import unittest

import common


sep = os.path.sep
src_path = sep.join(common.__file__.split(sep)[:-2] + ['gpu', 'src', ''])


def gpu_device_list(print_info=True):
    """
    Return the list of the GPU devices
    """

    platforms = cl.get_platforms()
    for i, platform in enumerate(platforms):
        if print_info:
            print('[%d] Platform : %s (%s)' % (i, \
                    platform.get_info(cl.platform_info.NAME), \
                    platform.get_info(cl.platform_info.VERSION)) )

        vendor = platform.get_info(cl.platform_info.VENDOR)
        if 'NVIDIA' in vendor or 'Advanced Micro Devices' in vendor:
            platform = platforms[i]
            if print_info:
                print('Platform [%d] is selected.' % i)
            break

    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)

    if print_info:
        print_gpu_info(gpu_devices)

    return gpu_devices



def print_gpu_info(devices):
    gpu_groups = {}
    for device in devices:
        name = device.get_info(cl.device_info.NAME)
        if not gpu_groups.has_key(name):
            gpu_groups[name] = {'count':1}
            gpu_groups[name]['compute units'] = \
                    device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            gpu_groups[name]['global mem size'] = \
                    device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            gpu_groups[name]['local mem size'] = \
                    device.get_info(cl.device_info.LOCAL_MEM_SIZE)
            gpu_groups[name]['constant mem size'] = \
                    device.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE)
        else:
            gpu_groups[name]['count'] += 1

    for name, props in gpu_groups.items():
        print('Device : %d GPU' % props['count'])
        print('  name: %s' % name)
        print('  compute units: %d' % props['compute units'])
        print('  global mem size: %1.2f %s' % \
                common.binary_prefix_nbytes(props['global mem size']) )
        print('  local mem size: %1.2f %s' % \
                common.binary_prefix_nbytes(props['local mem size']) )
        print('  constant mem size: %1.2f %s' % \
                common.binary_prefix_nbytes(props['constant mem size']) )
        print('')



def get_optimal_gs(device):     # global_work_size
    """
    Return the optimal global-work-size

    Use for the Nvidia and AMD ATI
    """

    vendor = device.get_info(cl.device_info.VENDOR)

    if 'NVIDIA' in vendor:
        warp_size = 32
        max_resident_warp_dict = {
                '1.0': 24, '1.1': 24,
                '1.2': 32, '1.3': 32,
                '2.0': 48}
        compute_capability = \
                str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV)) + \
                '.' + \
                str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV))

        max_resident_warp = max_resident_warp_dict[compute_capability]
        max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        optimal_gs = max_compute_units * max_resident_warp * warp_size

    elif 'Advanced Micro Devices' in vendor:
        wavefront_size = 64
        name = device.get_info(cl.device_info.NAME)
        max_wavefront_per_gpu = { \
                'Zacate': 192, \
                'Ontario': 192, \
                'Cedar': 192, \
                'Seymour': 192, \
                'Caicos': 192, \
                'Turks': 248, \
                'Whistler': 248, \
                'Redwood': 248, \
                'Juniper': 248, \
                'Barts': 496, \
                'Blackcomb': 496, \
                'Cypress': 496, \
                'Cayman': 512, \
                'Hemlock': 992}[name]
        optimal_gs = max_wavefront_per_gpu * wavefront_size

    return optimal_gs



def macro_replace_list(pt0, pt1):
    """
    Return the replace string list correspond to macro

    This is used to generate the opencl kernel from the template.
    """

    common.check_type('pt0', pt0, (list, tuple), int)
    common.check_type('pt1', pt1, (list, tuple), int)

    x0, y0, z0 = pt0
    x1, y1, z1 = pt1

    snx = abs(x1 - x0) + 1
    sny = abs(y1 - y0) + 1
    snz = abs(z1 - z0) + 1

    nmax = snx * sny * snz
    xid, yid, zid = x0, y0, z0

    if x0 == x1 and y0 == y1 and z0 == z1:
        pass

    elif x0 != x1 and y0 == y1 and z0 == z1:
        xid = '(gid + %d)' % x0
	
    elif x0 == x1 and y0 != y1 and z0 == z1:
        yid = '(gid + %d)' % y0
	
    elif x0 == x1 and y0 == y1 and z0 != z1:
        zid = '(gid + %d)' % z0
	
    elif x0 != x1 and y0 != y1 and z0 == z1:
        xid = '(gid/%d + %d)' % (sny, x0)
        yid = '(gid%%%d + %d)' % (sny, y0)
	
    elif x0 == x1 and y0 != y1 and z0 != z1:
        yid = '(gid/%d + %d)' % (snz, y0)
        zid = '(gid%%%d + %d)' % (snz, z0)
	
    elif x0 != x1 and y0 == y1 and z0 != z1:
        xid = '(gid/%d + %d)' % (snz, x0)
        zid = '(gid%%%d + %d)' % (snz, z0)
	
    elif x0 != x1 and y0 != y1 and z0 != z1:
        xid = '(gid/%d + %d)' % (sny*snz, x0)
        yid = '((gid/%d)%%%d + %d)' % (snz, sny, y0)
        zid = '(gid%%%d + %d)' % (snz, z0)
	
    return [str(nmax), str(xid), str(yid), str(zid)]




class TestFunctions(unittest.TestCase):
    def test_get_optimal_gs(self):
        gpu_devices = gpu_device_list()
        print('Optimal Gs = %d' % get_optimal_gs(gpu_devices[0]))



if __name__ == '__main__':
    unittest.main()
