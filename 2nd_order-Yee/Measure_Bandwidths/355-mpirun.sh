#!/bin/sh

mpirun -np 3 --machinefile machines 355-measure-mpi_persistent_nonblocking.py
