#!/bin/sh

rm *.exe
gcc -O3 -lm -lhdf5 010-cpu.c -o 010-cpu.exe

gcc -O3 -lm -lhdf5 011-cpu.c -o 011-cpu.exe

gcc -lm -lhdf5 015-cpu-non_ch.c -o 015-cpu-non_ch.exe

gcc -lm -lhdf5 016-cpu-non_ch.c -o 016-cpu-non_ch.exe

gcc -O2 -lm -lhdf5 -msse -fopenmp 020-cpu-sse-omp.c -o 020-cpu-sse-omp.exe

gcc -O3 -lm -lhdf5 -msse -fopenmp 021-cpu-sse-omp.c -o 021-cpu-sse-omp.exe

gcc -O2 -lm -lhdf5 -msse -fopenmp 025-cpu-non_ch-sse-omp.c -o 025-cpu-non_ch-sse-omp.exe

gcc -O3 -lm -lhdf5 -msse 030-cpu-sse.c -o 030-cpu-sse.exe

gcc -O3 -lm -lhdf5 -msse 031-cpu-sse.c -o 031-cpu-sse.exe

gcc -lm -lhdf5 -msse 035-cpu-non_ch-sse.c -o 035-cpu-non_ch-sse.exe

gcc -lm -lhdf5 -msse 036-cpu-non_ch-sse.c -o 036-cpu-non_ch-sse.exe

gcc -lm -lhdf5 -msse -fopenmp 038-cpu-non_ch-sse-omp.c -o 038-cpu-non_ch-sse-omp.exe

export OMP_NUM_THREADS=4
