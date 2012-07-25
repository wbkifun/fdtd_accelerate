#!/usr/bin/env python

import numpy as np
import subprocess as sp
import sys


nxs = range(96, 480+1, 32)	# nx**3, 30.38 MiB ~ 3.71 GiB
#nxs = [96, 128]
dts = np.zeros(len(nxs))
tmax = 1000

for i, nx in enumerate(nxs):
	cmd = './322-get_dt_gpu.py %d %d' % (nx, tmax)
	proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
	stdout, stderr = proc.communicate()
	if stderr != '': print('stderr :\n %s\n' % stderr)
	#print stdout

	dts[i] = float(stdout)
	mcells = ( nx**3 * tmax / dts[i] ) / 1e6
	print('nx = %d, dt = %f, %1.3f MCells/s' % (nx, dts[i], mcells))


# Save as h5
import pyopencl as cl
platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
gpu_name = gpu_devices[0].get_info(cl.device_info.NAME)
print gpu_name

import h5py as h5
h5_path = './capability_fdtd3d.h5'
f = h5.File(h5_path, 'a')
if 'tmax' not in f.attrs.keys():
	f.attrs['tmax'] = tmax
elif f.attrs['tmax'] != tmax:
	print('The \'tmax\' value is not matched in h5 file\'s attrs')
	sys.exit()

if 'gpu' not in f.keys():
	f.create_group('gpu')
if gpu_name not in f['gpu'].keys():
	f['gpu'].create_group(gpu_name)
f['gpu'][gpu_name].create_dataset('nx', data=np.array(nxs))
f['gpu'][gpu_name].create_dataset('dt', data=dts)
f.close()
