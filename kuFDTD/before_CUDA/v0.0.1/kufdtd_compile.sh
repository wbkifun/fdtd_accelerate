#!/bin/bash
rm kufdtd_3d_core.o kufdtd_3d_core.so
gcc -O3 -fpic -g -I. -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -c kufdtd_3d_core.c -o kufdtd_3d_core.o
gcc -shared -o kufdtd_3d_core.so kufdtd_3d_core.o
rm kufdtd_3d_cpml_core.o kufdtd_3d_cpml_core.so
gcc -O3 -fpic -g -I. -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -c kufdtd_3d_cpml_core.c -o kufdtd_3d_cpml_core.o
gcc -shared -o kufdtd_3d_cpml_core.so kufdtd_3d_cpml_core.o
