#!/usr/bin/env python

from scipy.io.numpyio import fread

def read_scbin_2d(scbin_path, Nx, Ny):
	fd = open(scbin_path, 'rb')
	array = fread(fd, Nx*Ny, 'f')
	array = array.reshape(Nx, Ny)
	fd.close()

	return array


def read_scbin_1d(scbin_path, Nx):
	fd = open(scbin_path, 'rb')
	array = fread(fd, Nx, 'f')
	fd.close()

	return array
