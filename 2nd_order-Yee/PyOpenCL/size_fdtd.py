#!/usr/bin/env python

from utils import print_nbytes


for i in xrange(1, 16):
	nx = i * 32

	head_str = 'i = %d, nx = %d, ' % (i, nx)
	print_nbytes(head_str, nx, nx, nx, 9)
