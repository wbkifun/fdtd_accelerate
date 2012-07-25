#!/bin/sh

mpirun -np 3 --machinefile machines 352-measure-mpi.py
