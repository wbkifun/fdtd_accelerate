import pycuda.driver as cuda
import os
import unittest

import common


sep = os.path.sep
src_path = sep.join(common.__file__.split(sep)[:-2] + ['gpu', 'src', ''])

cuda.init()


def print_gpu_info():
    print('CUDA version : %d.%d.%d' % cuda.get_version())
    gpu0 = cuda.Device(0)
    ngpu = gpu0.count()

    gpu_list = [cuda.Device(i) for i in range(ngpu)]
    gpu_groups = {}
    for gpu in gpu_list:
        name = gpu.name()
        if not gpu_groups.has_key(name):
            gpu_groups[name] = {'count' : 1}
            gpu_groups[name]['compute capability'] = gpu.compute_capability()
            gpu_groups[name]['global mem size'] = gpu.total_memory()
            gpu_groups[name]['multiprocessor'] = \
                    gpu.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        else:
            gpu_groups[name]['count'] += 1

    for name, props in gpu_groups.items():
        print('Device : %d GPU' % props['count'])
        print('  name: %s' % name)
        print('  compute capability: %d.%d' % props['compute capability'])
        print('  multiprocessor: %d' % props['multiprocessor'])
        print('  global mem size: %1.2f %s' % \
                common.binary_prefix_nbytes(props['global mem size']) )
        print('')



def get_optimal_gs(device, block_size):
    """
    Return the optimal grid size
    """

    attr_dict = device.get_attributes()
    sm = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    max_threads_per_sm = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
    
    return sm * max_threads_per_sm / block_size



def macro_replace_list(pt0, pt1):
    """
    Return the replace string list correspond to macro

    This is used to generate the cuda kernel from the template.
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
        gpu0 = cuda.Device(0)
        print('Optimal Gs = %d' % get_optimal_gs(gpu0, 256))



if __name__ == '__main__':
    print_gpu_info()
    unittest.main()
