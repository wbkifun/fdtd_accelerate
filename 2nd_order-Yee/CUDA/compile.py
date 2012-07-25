#!/usr/bin/env python

import os, sys
import glob

exec_filename = sys.argv[1]
cu_filename = exec_filename + '.cu'
linkinfo_filename = exec_filename + '.linkinfo'

file_list = glob.glob(exec_filename+'.*')
if linkinfo_filename in file_list:
	os.remove(linkinfo_filename)
if exec_filename in file_list:
	os.remove(exec_filename)

#cmd = 'nvcc -O3 -std=c99 -lm -lhdf5 %s -o %s' % (cu_filename, exec_filename)
#cmd = 'nvcc -O3 -lm -lhdf5 -deviceemu %s -o %s' % (cu_filename, exec_filename)
#cmd = 'nvcc -O3 -lm -lhdf5 %s -o %s' % (cu_filename, exec_filename)
#cmd = 'nvcc -use_fast_math -lm -lhdf5 %s -o %s' % (cu_filename, exec_filename)
#cmd = 'nvcc -cubin -use_fast_math -lm -lhdf5 %s -o %s' % (cu_filename, exec_filename)
#cmd = 'nvcc --ptxas-options=-v -lm -lhdf5 %s -o %s' % (cu_filename, exec_filename)
#cmd = 'nvcc -deviceemu -lm -lhdf5 %s -o %s' % (cu_filename, exec_filename)
cmd = 'nvcc -lhdf5 %s -o %s' % (cu_filename, exec_filename)
print cmd
os.system(cmd)

'''
cmd = './%s' % (exec_filename)
print cmd
os.system(cmd)
'''
