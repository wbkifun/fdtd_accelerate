#!/usr/bin/env python

import os

#c_file = 'dielectric-naive.c'
#c_file = 'dielectric-omp.c'
c_file = 'dielectric-sse-omp.c'
#c_file = 'dielectric-sse-omp-stream.c'
so_file = 'dielectric.so'

#cmd = 'gcc -fpic -I/usr/include/python2.6 -I/usr/lib/python2.6/site-packages/numpy/core/include -shared %s -o %s' %(c_file, so_file)
#cmd = 'gcc -O3 -fpic -fopenmp -I/usr/include/python2.6 -I/usr/lib/python2.6/site-packages/numpy/core/include -shared %s -o %s' %(c_file, so_file)
cmd = 'gcc -O3 -fpic -fopenmp -msse -I/usr/include/python2.6 -I/usr/lib/python2.6/site-packages/numpy/core/include -shared %s -o %s' %(c_file, so_file)

#cmd = 'gcc -O3 -fpic -fopenmp -msse -fprefetch-loop-arrays -I/usr/include/python2.6 -I/usr/lib/python2.6/site-packages/numpy/core/include -shared %s -o %s' %(c_file, so_file)

print cmd

os.system(cmd)
