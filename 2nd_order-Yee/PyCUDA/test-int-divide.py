#!/usr/bin/env python

nx, ny, nz = 4, 5, 6

for idx in xrange(nx*ny*nz):
	i = idx/(ny*nz)
	j = idx/nz
	k = idx%nz

	j2 = (idx - i*ny*nz)/nz
	k2 = idx - i*ny*nz - j2*nz
	print i,j,k,'\t',j2,k2
