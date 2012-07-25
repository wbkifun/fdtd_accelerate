import sys
sys.path.append('/home/kifang')

import numpy as np
import pyopencl as cl
from kemp.fdtd3d import common_gpu
from kemp.fdtd3d.gpu import Fields, GetFields, Npml

nx, ny, nz = 100, 110, 128
gpu_id = 0

gpu_devices = common_gpu.get_gpu_devices()
context = cl.Context(gpu_devices)
device = gpu_devices[gpu_id]
fdtd = Fields(context, device, nx, ny, nz, coeff_use='')

fhosts = {}
pml = Npml(fdtd, -0.428571428571, 0.714285714286, 0.6, 0.2, 0.6, 0.2)

fhost = np.random.rand(nx, nz).astype(fdtd.dtype)
cl.enqueue_write_buffer(fdtd.queue, pml.pex, fhost)

pml.update_h()
fget = GetFields(fdtd, ['hz','hx'], (0, ny-1, 0), (nx-1, ny-1, nz-1))
fget.get_event().wait()
hz = fget.get_fields('hz')
hx = fget.get_fields('hx')

print fhost
print hz
print fhost.shape
print hz.shape

assert np.linalg.norm(fhost - hz) == 0
