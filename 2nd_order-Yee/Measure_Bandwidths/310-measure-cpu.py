#!/usr/bin/env python

import numpy as np
import subprocess as sp
import sys


nxs = range(96, 480+1, 32)	# nx**3, 30.38 MiB ~ 3.71 GiB
#nxs = [96, 128]
dts = np.zeros(len(nxs))
tmax = 1000

for i, nx in enumerate(nxs):
	cmd = './312-get_dt_cpu.py %d %d' % (nx, tmax)
	proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
	stdout, stderr = proc.communicate()
	if stderr != '': print('stderr :\n %s\n' % stderr)
	#print stdout

	dts[i] = float(stdout)
	mcells = ( nx**3 * tmax / dts[i] ) / 1e6
	print('nx = %d, dt = %f, %1.3f MCells/s' % (nx, dts[i], mcells))


# Save as h5
import h5py as h5
for line in open('/proc/cpuinfo'):
	if 'model name' in line:
		cpu_name0 = line[line.find(':')+1:-1]
		break
cpu_name = ''
for s in cpu_name0.split():
	cpu_name += (s + ' ')
print cpu_name

h5_path = './capability_fdtd3d.h5'
f = h5.File(h5_path, 'a')
if 'tmax' not in f.attrs.keys():
	f.attrs['tmax'] = tmax
elif f.attrs['tmax'] != tmax:
	print('The \'tmax\' value is not matched in h5 file\'s attrs')
	sys.exit()

if 'cpu' not in f.keys():
	f.create_group('cpu')
if cpu_name not in f['cpu'].keys():
	f['cpu'].create_group(cpu_name)
f['cpu'][cpu_name].create_dataset('nx_4T', data=np.array(nxs))
f['cpu'][cpu_name].create_dataset('dt_4T', data=dts)
f.close()
