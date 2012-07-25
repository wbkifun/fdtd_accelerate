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
    if len(platforms) > 1:
        for i, platform in enumerate(platforms):
            print('[%d] Platform : %s (%s)' % (i, \
                    platform.get_info(cl.platform_info.NAME), \
                    platform.get_info(cl.platform_info.VERSION)) )

            platform_number = raw_input('Choice [0]: ')
            if platform_number in ['', '0']:
                platform = platforms[0]
            else:
                platform = platforms[int(platform_number)]
    else:
        platform = platforms[0]

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

    Use only for a Nvidia GPU
    """

    warp_size = 32
    max_resident_warp_dict = {
            '1.0':24, '1.1':24,
            '1.2':32, '1.3':32,
            '2.0':48}
    compute_capability = \
            str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV)) + \
            '.' + \
            str(device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV))

    max_resident_warp = max_resident_warp_dict[compute_capability]
    max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)

    return max_compute_units * max_resident_warp * warp_size



class TestFunctions(unittest.TestCase):
    def test_get_optimal_gs(self):
        gpu_devices = gpu_device_list()
        print('Optimal Gs = %d' % get_optimal_gs(gpu_devices[0]))



if __name__ == '__main__':
    unittest.main()
