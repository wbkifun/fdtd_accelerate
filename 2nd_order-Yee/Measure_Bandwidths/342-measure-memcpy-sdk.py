#!/usr/bin/env python

import numpy as np
import subprocess as sp
import sys

def get_bandwidth(cmd, arr):
	#print(cmd)
	out, err = sp.Popen(cmd.split(), stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE).communicate('\n')
	if err != '': 
		print(err)
		sys.exit()
	else:
		lines = out.splitlines()
		sublines = lines[10:-7]
		for i, line in enumerate(sublines):
			nbytes, bandwidth = line.split()
			arr[i] = float(bandwidth)


# verify h5 file exist
import os
out, err = sp.Popen(['hostname'], stdout=sp.PIPE).communicate()
h5_path = './bandwidth_PCIE.%s.h5' % out.rstrip('\n')
if os.path.exists(h5_path):
	print('Error: File exist %s' % h5_path)
	sys.exit()


# Main
start = 2 * np.nbytes['float32'] * (3 * 32)**2	# 96, 73728 
end = 2 * np.nbytes['float32'] * (15 * 32)**2	# 480, 1843200 
increment = (end - start) / 16

f = np.zeros(17)
datas = {'nbytes':np.arange(start, end+1, increment),
		'dtod':f.copy(),
		'dtoh':{'pageable':f.copy(), 'pinned':f.copy()},
		'htod':{'pageable':f.copy(), 'pinned':f.copy()}}
		
cmd_path = '/home/kifang/NVIDIA_GPU_Computing_SDK/OpenCL/bin/linux/release/oclBandwidthTest '
cmd_range = '--mode=range --start=%d --end=%d --increment=%d ' % (start, end, increment)
for key, value in datas.items():
	if key == 'nbytes':
		pass
	elif key == 'dtod':
		cmd = cmd_path + cmd_range + '--dtod'
		get_bandwidth(cmd, value)
	else:
		for key2, value2 in value.items():
			cmd = cmd_path + cmd_range + '--memory=%s --%s' % (key2, key)
			get_bandwidth(cmd, value2)


# Save as h5
import h5py as h5
f = h5.File(h5_path, 'w')
for key, value in datas.items():
	if type(value) == np.ndarray:
		f.create_dataset(key, data=value)
	elif type(value) == dict:
		f.create_group(key)
		for key2, value2 in value.items():
			f[key].create_dataset(key2, data=value2)
