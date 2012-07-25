#!/usr/bin/env python

import os

nx, ny, nz = 240, 256, 256
size = nx*ny*nz*4

print 'dim (%d, %d, %d)' % (nx, ny, nz)
total_bytes = size
if total_bytes/(1024**3) == 0:
	print 'mem %1.2f MB' % ( float(total_bytes)/(1024**2) )
else:
	print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )


path = '/home/kifang/NVIDIA_GPU_Computing_SDK/C/bin/linux/release/bandwidthTest'
args = ' --dtod --mode=range --start=%d --end=%d --increment=%d' % (size-1024, size+1024, 1024)
cmd = path + args
print cmd
os.system(cmd)
