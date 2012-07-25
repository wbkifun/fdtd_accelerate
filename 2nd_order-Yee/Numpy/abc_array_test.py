#!/usr/bin/env python

import numpy as np

n = 2
ez = np.zeros(10)

for tstep in xrange(5):
	ez[:] = np.random.rand(10)
	abc_ez = np.zeros(2*n)

	ez[-1] = abc_ez[-1]
	abc_ez[1:][::-1] = abc_ez[:-1][::-1]
	abc_ez[0] = ez[-2]

	print(ez)
	print(abc_ez)
