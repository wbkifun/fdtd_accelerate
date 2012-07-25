#!/usr/bin/env python

import os, sys

cu_file = sys.argv[1] + '.cu'
exe_file = sys.argv[1]

cmd = 'rm %s' % (exe_file)
print cmd
os.system(cmd)

cmd = 'nvcc -Xcompiler -fopenmp -lgomp %s -o %s' % (cu_file, exe_file)
print cmd
os.system(cmd)
